# tajweed-server

The Sabaq tajweed-check server. One endpoint judges a recitation of
one verse: Quran Muaalem v3.2 (Hafs only) as the judge, a calibrated
acoustic duration check as a second opinion on separate madd, and
three-way gating (`correct` / `error` / `uncertain`) so a wrong
correction is never shown as a correction.

Repo: https://github.com/somilli/tajweed-server

## Layout

```
app/
  server.py     FastAPI: GET /health, POST /check
  engine.py     model load (bfloat16) + single-forward inference
                + CTC forced alignment + duration measurement
  reference.py  verse -> Uthmani + QPS + tajweed rule sites
  measure.py    QPS alignment, categorisation, Rule B gating,
                three-way combine, feedback strings
  config.py     calibrated thresholds, valid sets, QML advisory text
setup.sh        Ubuntu 24.04 install: venv, model download, API key,
                systemd service, Caddy + HTTPS, firewall
tajweed.service systemd unit (uvicorn on 127.0.0.1:8000)
Caddyfile       reverse-proxy template (__DOMAIN__ substituted)
requirements.txt pinned versions (torch 2.15.0, transformers 4.57.6,
                quran-muaalem 0.3.0, quran-transcript 0.6.4, ...)
SPEC.md         the API contract, the judging pipeline, the honest
                numbers, and the app-side obligations (QML advisory
                + free/hardship path + complete recitations only)
deploy.md       how to order the VPS, point the subdomain, install,
                verify, and run it day to day
tests/
  test_parser_regression.py  measurement vs the reference parser on
                all 159 qdat_bench clips (PASS: 4 known deviations)
  test_reference_sites.py    hand-checked sites for Fatiha 1:1-1:7,
                An-Naba 78:1, Al-Ma'idah 5:109
  test_server_smoke.py       boots the real server, checks real clips
                end to end (pred QPS + verdicts vs the offline run)
```

## Run it locally

```bash
python3 -m venv ~/.venvs/sabaq-ear-test   # if not already there
~/.venvs/sabaq-ear-test/bin/pip install -r requirements.txt
TAJWEED_MODEL_DIR=/path/to/muaalem-v3_2 \
  ~/.venvs/sabaq-ear-test/bin/python -m uvicorn app.server:app \
  --host 127.0.0.1 --port 8000
```

The model dir is a standard Hugging Face layout
(`huggingface-cli download obadx/muaalem-model-v3_2 --local-dir
/path/to/muaalem-v3_2`). First load takes ~10 s and ~2.2 GB RAM;
inference peaks at ~3.2 GB, so a 4 GB machine is the floor.

Audio is decoded with ffmpeg (not librosa's loader — libsndfile
cannot open m4a/AAC). `TAJWEED_API_KEY` unset means no auth (dev
only); set it for anything real.

## Tests

From this directory, with the tajweed venv:

```bash
~/.venvs/sabaq-ear-test/bin/python tests/test_parser_regression.py
~/.venvs/sabaq-ear-test/bin/python tests/test_reference_sites.py
~/.venvs/sabaq-ear-test/bin/python tests/test_server_smoke.py  # ~1 min
```

The smoke test boots uvicorn on port 8001, checks two 5:109 clips
(pred QPS and model verdicts must equal the offline run, duration
verdicts present), one m4a clip, and the 401/400 paths, then kills
the server. All three passed Nov 12, 2026.

## Deploy

See `deploy.md`. Short version: OVHcloud VPS-1 in Virginia,
$5.35/month, Ubuntu 24.04, A record `tajweed.saheldev.group`, then
`sudo bash setup.sh tajweed.saheldev.group`. Save the API key it
prints once.

## License note

Quran Muaalem is **QML 1.0** (not the MIT tag on Hugging Face).
Commercial use is allowed with an in-app AI-advisory notice and a
free-access path for those who cannot pay — see SPEC.md. Whether
QML 1.0 permits shipping a fine-tuned derivative is unknown; check
the LICENSE text before distributing fine-tuned weights.
quran-transcript is MIT; Quran text is Tanzil CC-BY 3.0.
