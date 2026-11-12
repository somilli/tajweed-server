"""Measure the predicted QPS against the reference sites, and gate the
verdicts three ways (correct / error / uncertain).

Measurement: an edit-distance (Levenshtein) alignment between the
reference QPS and the predicted QPS locates each site in the
prediction; the run is then counted in full. A dropped word shows up
as deletions, a wrong letter as a substitution — so the measurement
stays on the right site even when the model drops whole words (the
sequential anchor search it replaced matched later occurrences
instead and skipped the real sites: 629/1431 mismatches on the 159
qdat clips). A site whose reference char was deleted in the prediction
is unparseable -> uncertain, never an error. Run length 0 is a real
measurement (the learner dropped the madd / said it plain) -> error,
because 0 is in no valid set.

Categorisation (from qdat_parse.categorize, calibrated on the 5:109
annotations):
  * noon/meem shaddah run >= 4 -> 1 (full ghunnah), run 2-3 -> 0.
  * ikhfaa ں run >= 3 -> 2 (full), 1-2 -> 1 (partial), 0 -> 0 (plain).
  * madd runs and qalqalah are used as measured.

Gating (Layer 3 of the tajweed solution):
  * Non-separate-madd rules: the model verdict alone.
  * Separate madd: the model verdict + the Rule B duration verdict.
      model error + duration error     -> error (both agree)
      model error + duration correct   -> uncertain (duration doubts it)
      model error + duration uncertain -> error (model alone, as before)
      model correct + duration error   -> uncertain (duration suspects a
                                         miss — this is where Rule B's
                                         18.4% detection lives, as soft
                                         feedback, never a hard correction)
      model correct + duration correct -> correct
      model correct + duration uncertain -> uncertain
      model uncertain                  -> uncertain
  The duration check is a second opinion, never the judge.
"""
from . import config

VALID = config.VALID


def categorize(site, measured):
    """Raw run length -> the categorical value the valid set is checked
    against. Returns None if the site is unparseable."""
    if measured is None:
        return None
    kind = site["kind"]
    if kind in ("noon_moshaddadah", "meem_moshaddadah"):
        return 1 if measured >= 4 else 0
    if kind == "noon_mokhfah":
        if measured >= 3:
            return 2
        if measured >= 1:
            return 1
        return 0
    return measured


def model_verdict(site, measured):
    """'correct' / 'error' / 'uncertain' from the model alone."""
    cat = categorize(site, measured)
    if cat is None:
        return "uncertain"
    return "correct" if cat in VALID[site["kind"]] else "error"


def rule_b_verdict(beats):
    """Rule B on the CTC path-span beats (separate madd only).
    delta 0.5, grey 0.75, valid {2,4,5}. beats None -> uncertain."""
    if beats is None:
        return "uncertain"
    valid = VALID["separate_madd"]
    dmin = min(abs(beats - v) for v in valid)
    if dmin <= config.RULE_B_DELTA:
        return "correct"
    if dmin <= config.RULE_B_DELTA + config.RULE_B_GREY:
        return "uncertain"
    return "error"


def combine(model_v, duration_v):
    """Three-way combine (duration_v is None for non-separate rules)."""
    if duration_v is None:
        return model_v
    if model_v == "error" and duration_v == "error":
        return "error"
    if model_v == "error" and duration_v == "correct":
        return "uncertain"
    if model_v == "error":
        return "error"
    if model_v == "correct" and duration_v == "error":
        return "uncertain"
    if model_v == "correct" and duration_v == "correct":
        return "correct"
    return "uncertain"


def _levenshtein_map(ref, pred):
    """Align ref (no-space) to pred (no-space) by edit distance.

    Returns ref_to_pred: a list of len(ref); ref_to_pred[i] is the index
    in pred that ref[i] aligns to, or None if ref[i] is deleted in the
    prediction. A dropped word shows up as deletions; a wrong letter as
    a substitution; an extra sound as an insertion. This is what makes
    the measurement robust when the model drops whole words — the old
    sequential anchor search matched later occurrences instead and
    skipped the real sites (629/1431 mismatches on the 159 clips).
    """
    n, m = len(ref), len(pred)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            # Substitution costs 2, insert/delete 1: a wrong letter is
            # better explained as "not said" (delete) + "something else
            # said" (insert) than as a free relabelling. With cost 1 the
            # DP relabelled ں->ن and mapped runs to the wrong word.
            cost = 0 if ri == pred[j - 1] else 2
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)
    ref_to_pred = [None] * n
    i, j = n, m
    while i > 0 or j > 0:
        if (i > 0 and j > 0 and ref[i - 1] == pred[j - 1]
                and dp[i][j] == dp[i - 1][j - 1]):
            ref_to_pred[i - 1] = j - 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 2:
            ref_to_pred[i - 1] = j - 1  # substitution
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ref_to_pred[i - 1] = None  # deletion
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            j -= 1  # insertion
        elif i > 0 and j > 0:
            ref_to_pred[i - 1] = j - 1
            i -= 1
            j -= 1
        elif i > 0:
            i -= 1
        else:
            j -= 1
    return ref_to_pred


def _count_run(p, start, ch):
    """Length of the run of ch starting at `start` (0 if p[start] != ch)."""
    if start < 0 or start >= len(p) or p[start] != ch:
        return 0
    j = start
    while j < len(p) and p[j] == ch:
        j += 1
    return j - start


def _measure_at(p, k, site):
    """Measure a site whose run should start at pred position k.

    k points at the run's first char (mapped or fallback). The 2 chars
    before k must be the site's anchor — if they are not, the alignment
    drifted to the wrong word, so look for the anchor in a small window
    around k and measure there; if it is nowhere nearby, the site is
    unparseable (None). Returns the run length, 1/0 for qalqalah, or
    None.
    """
    anchor = site.get("anchor") or ""
    ch = site["run_char"]
    if anchor and k > 0:
        # the anchor must sit immediately before k
        lo = max(0, k - len(anchor))
        if p[lo:k] != anchor[- (k - lo):]:
            # drifted: search the anchor in a +/-12 window around k
            w0, w1 = max(0, k - 12), min(len(p), k + 12)
            best = None
            start = w0
            while True:
                i = p.find(anchor, start, w1)
                if i < 0:
                    break
                if best is None or abs(i - k) < abs(best - k):
                    best = i
                start = i + 1
            if best is None:
                if site["kind"] == "qalqalah":
                    # the local context is too mangled to place the
                    # letter precisely: fall back to the tail check
                    # (qdat_parse's own rule) — an echo marker within
                    # a few chars of the expected position counts.
                    return 1 if "ڇ" in p[k:k + 6] else 0
                return None
            k = best + len(anchor)
    if site["kind"] == "qalqalah":
        if k >= len(p) or p[k] != ch:
            return 0
        return 1 if (k + 1 < len(p) and p[k + 1] == "ڇ") else 0
    # the run: expand back to its start (the alignment may have landed
    # inside it), then count forward.
    if k >= len(p):
        return 0
    if p[k] != ch:
        return 0
    start = k
    while start > 0 and p[start - 1] == ch:
        start -= 1
    return _count_run(p, start, ch) if p[start] == ch else 0


def _run_start_after(mapping, off, pred_len):
    """Pred position where the run starting at ref offset `off` should
    begin, when ref[off] itself was deleted: just after the last mapped
    ref char before it. None if nothing before it maps (the site's
    words were not transcribed at all)."""
    for j in range(off - 1, -1, -1):
        if mapping[j] is not None:
            return mapping[j] + 1
    return None


def measure_predicted(pred_qps, ref_nospace, sites):
    """Per-site raw measurements from the predicted (no-space) QPS.

    Each reference site is located in the prediction through the
    edit-distance alignment (with the anchor check in _measure_at
    catching alignment drift); the run is then counted in full. Returns
    {site_id: measured} where measured is the run length (int), 1/0 for
    qalqalah, or None if the site cannot be located at all (unparseable
    -> uncertain, never an error). A run length of 0 is a real
    measurement (the learner dropped the madd / said it plain, or the
    site sits past the end of what was transcribed) -> error, because 0
    is in no valid set.
    """
    p = pred_qps.replace(" ", "")
    r = ref_nospace.replace(" ", "")
    mapping = _levenshtein_map(r, p)
    out = {}
    for site in sites:
        off = site["nospace_offset"]
        k = mapping[off] if 0 <= off < len(mapping) else None
        if k is None:
            # the run's first char was not produced: locate where it
            # should have started (just after the last mapped char
            # before it). None only if nothing before it maps.
            k = _run_start_after(mapping, off, len(p))
            if k is None:
                out[site["id"]] = None
                continue
        out[site["id"]] = _measure_at(p, k, site)
    return out


def _valid_str(valid):
    return ", ".join(str(v) for v in valid)


def feedback(site, verdict, measured, beats=None):
    """One plain line for the app to show for this site."""
    kind = site["kind"]
    if verdict == "correct":
        return "Good — that one was right."
    if verdict == "uncertain":
        return "Not sure on this one — try it again, or ask your teacher."
    # error
    if kind in ("madd_normal", "separate_madd", "madd_muttasel",
                "madd_lazim", "madd_aared"):
        held = measured if measured is not None else "?"
        if kind == "madd_aared":
            return (f"Held {held} beats — at a stop it should be "
                    f"{_valid_str(site['valid'])} beats.")
        return (f"Held {held} beats — it should be "
                f"{_valid_str(site['valid'])} beats.")
    if kind in ("noon_moshaddadah", "meem_moshaddadah"):
        letter = "noon" if kind == "noon_moshaddadah" else "meem"
        return (f"The doubled {letter} needs a full ghunnah — a nasal "
                f"sound held for 2 beats.")
    if kind == "noon_mokhfah":
        return "The noon here should be hidden (ikhfaa) — it sounded plain."
    if kind == "qalqalah":
        return "This letter needs its echo (qalqalah)."
    return "That one was not quite right — try it again."
