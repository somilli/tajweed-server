"""Smoke test: boot the real server and check real clips end to end.

* The server's predicted QPS must equal the offline run's pred QPS for
  the same clip (the engine's single forward must reproduce the
  high-level Muaalem call exactly).
* The server's model verdicts must equal the offline measurement on
  that pred QPS.
* The duration check (Rule B) must produce a verdict for the separate
  madd site.
* m4a decoding, auth (401), and bad-verse (400) behaviour.

Run from tajweed-server/ with the tajweed venv:
  python tests/test_server_smoke.py
Starts uvicorn on port 8001, kills it afterwards. Takes ~2-3 min
(model load ~11 s + 3 inferences).
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import measure, reference  # noqa: E402

BASE = os.path.expanduser(
    "~/workspace/goals/ai-project-company/files/sabaq-ear-test")
QDAT = os.path.join(BASE, "qdat-bench")
PORT = 8001
URL = f"http://127.0.0.1:{PORT}"
KEY = "smoke-test-key"
ENV = dict(os.environ,
           TAJWEED_MODEL_DIR=os.path.join(BASE, "models/muaalem-v3_2"),
           TAJWEED_API_KEY=KEY,
           TAJWEED_PORT=str(PORT))


def http_json(method, path, headers=None):
    req = urllib.request.Request(URL + path, method=method,
                                 headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def curl_check(audio_path, surah, ayah, key=KEY):
    cmd = ["curl", "-s", "-w", "\n%{http_code}",
           "-F", f"file=@{audio_path}",
           "-F", f"surah={surah}", "-F", f"ayah={ayah}"]
    if key is not None:
        cmd += ["-H", f"X-API-Key: {key}"]
    cmd += [URL + "/check"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    body, _, code = out.stdout.rpartition("\n")
    return int(code), json.loads(body) if body.strip() else {}


def main():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.server:app",
         "--host", "127.0.0.1", "--port", str(PORT), "--workers", "1"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    failures = []
    try:
        # wait for health (model load ~11 s)
        for _ in range(60):
            try:
                code, body = http_json("GET", "/health")
                if code == 200 and body.get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            print("FAIL — server did not become healthy")
            sys.exit(1)
        print(f"health: {body}")

        res = json.load(open(os.path.join(QDAT, "results-qdat.json")))
        ref = reference.build_reference(5, 109)

        for oid in ("s0_1", "s10_6"):
            wav = os.path.join(QDAT, "wav16k", oid + ".wav")
            code, resp = curl_check(wav, 5, 109)
            assert code == 200, f"{oid}: HTTP {code} {resp}"
            offline_pred = res["samples"][oid]["pred_phonemes"]
            print(f"\n{oid}: rtf={resp['rtf']} "
                  f"summary={resp['summary']}")
            if resp["predicted_qps"] != offline_pred:
                failures.append(
                    f"{oid}: pred QPS differs from the offline run\n"
                    f"  server : {resp['predicted_qps']}\n"
                    f"  offline: {offline_pred}")
            # model verdicts must match the offline measurement
            off_meas = measure.measure_predicted(
                offline_pred, ref["qps_nospace"], ref["sites"])
            by_id = {s["id"]: s for s in resp["sites"]}
            for site in ref["sites"]:
                got = by_id[site["id"]]
                want = measure.model_verdict(site, off_meas[site["id"]])
                if got["model_verdict"] != want:
                    failures.append(
                        f"{oid} {site['id']}: model verdict "
                        f"{got['model_verdict']} != offline {want}")
            # duration check present on every separate madd site
            # (5:109 has two: ماذا + لنا)
            sep = [s for s in resp["sites"]
                   if s["rule"] == "separate_madd"]
            assert len(sep) == 2, f"{oid}: expected 2 separate sites"
            for s in sep:
                if s["duration_verdict"] not in (
                        "correct", "error", "uncertain"):
                    failures.append(
                        f"{oid}: no duration verdict on separate "
                        f"({s['word']})")
                print(f"  separate ({s['word']}): "
                      f"model={s['model_verdict']} "
                      f"duration={s['duration_verdict']} "
                      f"beats={s['duration_beats']} "
                      f"final={s['verdict']}")
            # summary consistency
            s = resp["summary"]
            assert (s["n_error"] + s["n_uncertain"] + s["n_correct"]
                    == s["n_sites"] == len(resp["sites"])), oid
            assert resp["advisory_en"] and resp["advisory_ar"], oid

        # m4a decoding (Sory's clip 4, An-Naba 78:1)
        m4a = os.path.expanduser("~/workspace/user/files/4.m4a")
        code, resp = curl_check(m4a, 78, 1)
        assert code == 200, f"m4a: HTTP {code} {resp}"
        print(f"\nm4a (78:1): rtf={resp['rtf']} summary={resp['summary']} "
              f"sites={len(resp['sites'])}")

        # auth + validation
        code, _ = curl_check(os.path.join(QDAT, "wav16k", "s0_1.wav"),
                             5, 109, key="wrong")
        assert code == 401, f"bad key: HTTP {code}"
        code, _ = curl_check(os.path.join(QDAT, "wav16k", "s0_1.wav"),
                             5, 109, key=None)
        assert code == 401, f"no key: HTTP {code}"
        code, resp = curl_check(os.path.join(QDAT, "wav16k", "s0_1.wav"),
                                999, 1)
        assert code == 400, f"bad verse: HTTP {code}"
        print("auth/validation: 401/401/400 as expected")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("\nPASS — server smoke test")


if __name__ == "__main__":
    main()
