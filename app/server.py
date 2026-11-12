"""Sabaq tajweed-check server (FastAPI).

POST /check  — multipart: audio file + surah + ayah.
  Returns per-site verdicts (correct / error / uncertain) from the
  Muaalem judge + the Rule B duration second opinion (separate madd
  only), with the three-way gating from measure.py.
GET  /health — model loaded, uptime, version.

One inference at a time (the model is CPU; a second request waits up to
INFER_LOCK_TIMEOUT_S, then gets 503). Audio is decoded to 16 kHz mono
and rejected over 30 s (Muaalem's limit) or 5 MB.

Auth: X-API-Key header, checked against TAJWEED_API_KEY when it is set.
No key set = no auth (dev only; setup.sh always generates one).
"""
import os
import subprocess
import tempfile
import threading
import time

import librosa
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from . import config, engine as eng, measure, reference

app = FastAPI(title="Sabaq tajweed check",
              version="1.0.0")
_started = time.time()
_engine = None
_lock = threading.Lock()


@app.on_event("startup")
def _startup():
    global _engine
    _engine = eng.load_model()


@app.get("/health")
def health():
    return {
        "status": "ok" if _engine else "loading",
        "model": config.MODEL_NAME,
        "model_load_s": _engine["load_s"] if _engine else None,
        "uptime_s": round(time.time() - _started, 1),
        "auth_required": bool(config.API_KEY),
    }


@app.post("/check")
async def check(
    file: UploadFile = File(...),
    surah: int = Form(...),
    ayah: int = Form(...),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="bad or missing API key")
    if not (1 <= surah <= 114):
        raise HTTPException(status_code=400, detail="surah must be 1-114")
    try:
        ref = reference.build_reference(surah, ayah)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    data = await file.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="audio over 5 MB")
    # Decode via ffmpeg, not librosa's loader: librosa 1.x goes through
    # libsndfile first, which cannot open m4a/AAC at all (and the
    # audioread fallback does not trigger for it). ffmpeg handles
    # wav/m4a/mp3/ogg alike and is installed by setup.sh.
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    tmp_in = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp_in.write(data)
        tmp_in.close()
        tmp_out.close()
        conv = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", tmp_in.name,
             "-ac", "1", "-ar", "16000", "-f", "wav", tmp_out.name],
            capture_output=True, text=True, timeout=60)
        if conv.returncode != 0:
            raise HTTPException(
                status_code=400,
                detail="could not decode audio: "
                       f"{conv.stderr.strip()[:200]}")
        try:
            wave, _sr = librosa.load(tmp_out.name, sr=16000, mono=True)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"could not decode audio: {exc}") from exc
    finally:
        os.unlink(tmp_in.name)
        os.unlink(tmp_out.name)
    wave = np.asarray(wave, dtype=np.float32)
    duration_s = len(wave) / 16000
    if duration_s > config.MAX_AUDIO_SECONDS:
        raise HTTPException(status_code=400,
                            detail=f"audio is {duration_s:.1f}s; max 30s")
    if duration_s < 0.5:
        raise HTTPException(status_code=400, detail="audio too short")

    got = _lock.acquire(timeout=config.INFER_LOCK_TIMEOUT_S)
    if not got:
        raise HTTPException(status_code=503,
                            detail="server busy — try again in a minute")
    try:
        t0 = time.time()
        inf = eng.infer(_engine, wave)
        infer_s = time.time() - t0
        dur, dur_info = eng.duration_measure(
            _engine, inf["logp"], inf["frame_dur"],
            ref["qps_nospace"], ref["sites"])
        measured = measure.measure_predicted(
            inf["pred_qps"], ref["qps_nospace"], ref["sites"])
    finally:
        _lock.release()

    sites_out = []
    for site in ref["sites"]:
        m = measured.get(site["id"])
        mv = measure.model_verdict(site, m)
        d = dur.get(site["id"]) if site["kind"] == "separate_madd" else None
        beats = d["beats"] if d else None
        dv = measure.rule_b_verdict(beats) if d is not None else None
        verdict = measure.combine(mv, dv)
        sites_out.append({
            "id": site["id"],
            "rule": site["rule"],
            "label": site["label"],
            "word": site["word"],
            "expected": site["valid"],
            "model_measured": measure.categorize(site, m),
            "model_raw_run": m,
            "model_verdict": mv,
            "duration_beats": beats,
            "duration_verdict": dv,
            "verdict": verdict,
            "feedback": measure.feedback(site, verdict, m, beats),
        })
    summary = {
        "n_sites": len(sites_out),
        "n_error": sum(1 for s in sites_out if s["verdict"] == "error"),
        "n_uncertain": sum(1 for s in sites_out
                           if s["verdict"] == "uncertain"),
        "n_correct": sum(1 for s in sites_out if s["verdict"] == "correct"),
    }
    return JSONResponse({
        "verse": f"{surah}:{ayah}",
        "uthmani": ref["uthmani"],
        "duration_s": round(duration_s, 2),
        "inference_s": round(infer_s, 2),
        "rtf": round(infer_s / duration_s, 2) if duration_s else None,
        "model": config.MODEL_NAME,
        "reference_qps": ref["qps_nospace"],
        "predicted_qps": inf["pred_qps"],
        "duration_info": dur_info,
        "sites": sites_out,
        "summary": summary,
        "advisory_en": config.ADVISORY_EN,
        "advisory_ar": config.ADVISORY_AR,
    })
