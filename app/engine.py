"""Muaalem engine: one model forward per clip gives BOTH signals.

* pred QPS — the same greedy CTC decode the high-level Muaalem call
  uses (phonemes_level_greedy_decode on the phonemes logits), so the
  text is identical to what run_qdat.py recorded.
* CTC log-probs — for the forced alignment behind the duration check.

Model loading is the proven bfloat16 hand-build from run_qdat.py (the
stock constructor's fp32 load OOMs a 4 GB box; this one peaks ~2.2 GB
at load, ~3.2 GB under inference).

The forced-alignment code (Viterbi CTC, midpoint spans, harakah, pace)
is ported from ctc_duration.py, generalised: the alignment target is the
REFERENCE QPS (in production there is no expert transcript), and the
site positions come from reference.py instead of the hardcoded 5:109
regex anchors.
"""
import time

import numpy as np
import torch

from . import config

torch.set_num_threads(2)

BLANK = 0
DIACRITIC_IDS = {32, 33, 34}  # fatha, damma, kasra (short vowels)


def load_model(model_dir=None):
    """Load Muaalem (bfloat16, CPU). Returns the engine handle."""
    from quran_muaalem import Muaalem
    from quran_muaalem.modeling.modeling_multi_level_ctc import (
        Wav2Vec2BertForMultilevelCTC,
    )
    from quran_muaalem.modeling.multi_level_tokenizer import (
        MultiLevelTokenizer,
    )
    from transformers import AutoFeatureExtractor

    model_dir = model_dir or config.MODEL_DIR
    t0 = time.time()
    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(
        model_dir, dtype=torch.bfloat16
    )
    tokenizer = MultiLevelTokenizer(model_dir)
    processor = AutoFeatureExtractor.from_pretrained(model_dir)
    muaalem = Muaalem.__new__(Muaalem)
    muaalem.model = model
    muaalem.multi_level_tokenizer = tokenizer
    muaalem.processor = processor
    muaalem.device = "cpu"
    muaalem.dtype = torch.bfloat16
    model.to("cpu")
    model.eval()
    vocab = tokenizer.id_to_vocab["phonemes"]
    return {
        "muaalem": muaalem,
        "model": model,
        "processor": processor,
        "vocab": vocab,
        "vocab_inv": {v: k for k, v in vocab.items()},
        "load_s": round(time.time() - t0, 1),
    }


def infer(engine, wave):
    """One forward. Returns {pred_qps, logp, frame_dur, n_frames}."""
    from quran_muaalem.decode import phonemes_level_greedy_decode

    feats = engine["processor"]([wave], sampling_rate=16000,
                                return_tensors="pt")
    feats = {k: v.to("cpu", dtype=torch.bfloat16)
             for k, v in feats.items()}
    with torch.no_grad():
        outs = engine["model"](**feats, return_dict=False)[0]
    logits = outs["phonemes"][0]  # (T, 43)
    probs = torch.nn.functional.softmax(logits.float(), dim=-1)
    units = phonemes_level_greedy_decode(
        probs.unsqueeze(0), engine["vocab"])
    pred_qps = units[0].text
    logp = torch.nn.functional.log_softmax(
        logits.float(), dim=-1).numpy()
    frame_dur = (len(wave) / 16000) / logp.shape[0]
    return {
        "pred_qps": pred_qps,
        "logp": logp,
        "frame_dur": frame_dur,
        "n_frames": logp.shape[0],
    }


def ctc_forced_align(logp, target):
    """Viterbi CTC alignment of target ids to frames (from ctc_duration.py).

    Returns (spans, path, total_logp): spans = (start, end) frame ranges
    per target phoneme (end exclusive; None if 0 frames)."""
    T, V = logp.shape
    N = len(target)
    ext = [BLANK]
    for t in target:
        ext.append(t)
        ext.append(BLANK)
    S = len(ext)
    NEG = -1e30
    dp = np.full((T, S), NEG)
    bp = np.full((T, S), -1, dtype=np.int32)
    dp[0, 0] = logp[0, ext[0]]
    if S > 1:
        dp[0, 1] = logp[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            best, arg = dp[t - 1, s], s
            if s - 1 >= 0 and dp[t - 1, s - 1] > best:
                best, arg = dp[t - 1, s - 1], s - 1
            if (s - 2 >= 0 and ext[s] != BLANK and ext[s] != ext[s - 2]
                    and dp[t - 1, s - 2] > best):
                best, arg = dp[t - 1, s - 2], s - 2
            dp[t, s] = best + logp[t, ext[s]]
            bp[t, s] = arg
    end = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    path = [0] * T
    path[T - 1] = end
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    spans = []
    for i in range(N):
        s = 2 * i + 1
        frames = [t for t in range(T) if path[t] == s]
        if frames:
            spans.append((frames[0], frames[-1] + 1))
        else:
            spans.append(None)
    return spans, path, float(dp[T - 1, end])


def midpoint_spans(logp, target, spans, path):
    """Each phoneme's anchor is its peak frame; boundaries are midpoints
    between consecutive peaks (from ctc_duration.py)."""
    T = logp.shape[0]
    N = len(target)
    peaks = []
    for i in range(N):
        sp = spans[i]
        if sp is not None:
            a, b = sp
            peak = a + int(np.argmax(logp[a:b, target[i]]))
        else:
            s = 2 * i + 1
            after = [t for t in range(T) if path[t] > s]
            peak = after[0] if after else T - 1
        peaks.append(peak)
    out = []
    for i in range(N):
        start = 0 if i == 0 else (peaks[i - 1] + peaks[i]) // 2
        end = T if i == N - 1 else (peaks[i] + peaks[i + 1]) // 2
        out.append((start, max(end, start + 1)))
    return out, peaks


def duration_measure(engine, logp, frame_dur, ref_nospace, sites):
    """CTC path-span beats for the separate-madd sites.

    Alignment target = the reference QPS (what they should have said).
    Returns ({site_id: {beats, path_span_frames, conf, beats_mid}},
    {harakah_s, pace_gap_frames}). beats is None when the site cannot
    be measured (alignment gave it no frames, or no pace) — the caller
    turns that into 'uncertain', never an error.
    """
    inv = engine["vocab_inv"]
    s = ref_nospace.replace(" ", "")
    target, offsets = [], []
    for off, ch in enumerate(s):
        tid = inv.get(ch)
        if tid is not None:
            target.append(tid)
            offsets.append(off)
    spans, path, _score = ctc_forced_align(logp, target)
    mid, peaks = midpoint_spans(logp, target, spans, path)

    short_durs = [(mid[i][1] - mid[i][0]) * frame_dur
                  for i, tid in enumerate(target)
                  if tid in DIACRITIC_IDS]
    harakah = float(np.median(short_durs)) if short_durs else None
    gaps = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
    pace_gap = float(np.median(gaps)) if gaps else None

    out = {}
    for site in sites:
        if site["kind"] != "separate_madd":
            continue
        try:
            tpos = offsets.index(site["nospace_offset"])
        except ValueError:
            out[site["id"]] = {"beats": None}
            continue
        tid = target[tpos]
        j = tpos
        while j < len(target) and target[j] == tid:
            j += 1
        run = spans[tpos:j]
        if (pace_gap is None or pace_gap == 0
                or any(sp is None for sp in run)):
            out[site["id"]] = {"beats": None}
            continue
        p_span = run[-1][1] - run[0][0]
        conf = float(np.mean(
            [logp[peaks[k], tid] for k in range(tpos, j)]))
        start = mid[tpos][0]
        end = mid[j - 1][1]
        beats_mid = ((end - start) * frame_dur / harakah
                     if harakah else None)
        out[site["id"]] = {
            "beats": round(p_span / pace_gap, 2),
            "path_span_frames": p_span,
            "conf": round(conf, 2),
            "beats_mid": (round(beats_mid, 2)
                          if beats_mid is not None else None),
        }
    return out, {
        "harakah_s": round(harakah, 3) if harakah else None,
        "pace_gap_frames": pace_gap,
    }
