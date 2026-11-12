"""Regression test: the general parser vs qdat_parse.py on all 159
qdat_bench clips (5:109), site by site.

The parser is NOT expected to match qdat_parse exactly — qdat_parse
itself has limits (it cannot parse a site whose anchor the model
mangled, and it only covers 5:109). The test therefore reports a
breakdown and fails only if it changes from the known baseline:

* LOST (old valued, new None) — the parser loses information
  qdat_parse had. Must stay at the known set.
* DIFFER (both valued, disagree) — must stay at the known set.
* RECOVERED (old None, new valued) — the parser reads a site
  qdat_parse could not parse. Reported, not a failure.

Known baseline (Nov 12, 2026 — 4/1431 = 0.3%):
* s52_3 qalo_alif_len: LOST. The model dropped the first 6 words and
  mangled علم; the alignment deletes ref's قالوا entirely. The site
  becomes 'uncertain' instead of 'correct'. Accepted: one site in one
  heavily-mangled clip; the uncertainty is the honest answer there.
* s52_3 laa_alif_len: DIFFER old=2 new=0. Same mangled clip; the
  alignment maps لا's alif to the spurious run in the mangled علم.
  A false positive on one site. Accepted (same clip).
* s81_3 allam_alif_len: DIFFER old=0 new=2. Mine is MORE correct:
  the alif in علام was held 2 beats; qdat_parse's anchor matched a
  wrong لَ. Not a bug.
* s77_2 noon_moshaddadah_len: RECOVERED old=None new=1. Mine reads
  the ghunnah (run 4 -> full) where qdat_parse's anchor failed. Not
  a bug.

Run from tajweed-server/ with the tajweed venv:
  python tests/test_parser_regression.py
No model needed (reference + measurement only).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser(
    "~/workspace/goals/ai-project-company/files/sabaq-ear-test/qdat-bench"))

from qdat_parse import categorize, parse  # noqa: E402

from app import measure, reference  # noqa: E402

BASE = os.path.expanduser(
    "~/workspace/goals/ai-project-company/files/sabaq-ear-test/qdat-bench")

QDAT_SITES = [
    ("qalo_alif_len", 7, "ا"),
    ("qalo_waw_len", 7, "ۥ"),
    ("laa_alif_len", 8, "ا"),
    ("separate_madd", 10, "ا"),
    ("noon_moshaddadah_len", 11, "ن"),
    ("noon_mokhfah_len", 12, "ں"),
    ("allam_alif_len", 13, "ا"),
    ("madd_aared_len", 14, "ۥ"),
    ("qalqalah", 14, "ب"),
]
CAT_KEYS = {"noon_moshaddadah_len", "noon_mokhfah_len", "qalqalah"}

# (clip, key) -> expected category; see the module docstring.
KNOWN = {
    ("s52_3", "qalo_alif_len"): "LOST",
    ("s52_3", "laa_alif_len"): "DIFFER",
    ("s81_3", "allam_alif_len"): "DIFFER",
    ("s77_2", "noon_moshaddadah_len"): "RECOVERED",
}


def main():
    ref = reference.build_reference(5, 109)
    sites = ref["sites"]
    print(f"5:109 reference: {len(sites)} sites found")
    res = json.load(open(os.path.join(BASE, "results-qdat.json")))
    samples = res["samples"]
    mine = {}
    for s in sites:
        mine[(s["word_index"], s["run_char"])] = s

    found = {}
    n_checked = 0
    for oid, smp in samples.items():
        pred = smp["pred_phonemes"]
        old = categorize(parse(pred))
        new = measure.measure_predicted(pred, ref["qps_nospace"], sites)
        for key, wi, ch in QDAT_SITES:
            site = mine[(wi, ch)]
            raw_new = new[site["id"]]
            new_val = (measure.categorize(site, raw_new)
                       if key in CAT_KEYS else raw_new)
            old_val = old.get(key)
            n_checked += 1
            if old_val == new_val:
                continue
            if old_val is not None and new_val is None:
                cat = "LOST"
            elif old_val is None and new_val is not None:
                cat = "RECOVERED"
            else:
                cat = "DIFFER"
            found[(oid, key)] = (cat, old_val, new_val)

    print(f"checked {n_checked} site measurements across "
          f"{len(samples)} clips")
    for (oid, key), (cat, ov, nv) in sorted(found.items()):
        print(f"  {cat:9s} {oid} {key}: old={ov} new={nv}")
    ok = True
    for k, (cat, _ov, _nv) in found.items():
        if KNOWN.get(k) != cat:
            print(f"UNEXPECTED: {k} is {cat}, not in the known baseline")
            ok = False
    for k, cat in KNOWN.items():
        if k not in found:
            print(f"MISSING: known baseline case {k} ({cat}) no longer "
                  f"reproduces — the parser changed")
            ok = False
    if ok:
        print("PASS — parser matches the known baseline "
              f"({len(found)}/{n_checked} known deviations)")
    else:
        print("FAIL — the parser's behaviour changed from the baseline")
        sys.exit(1)


if __name__ == "__main__":
    main()
