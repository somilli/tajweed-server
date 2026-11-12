"""Spot-check the reference site-finder on verses whose tajweed is
known by hand (Al-Fatiha 1:1-1:7, An-Naba 78:1, Al-Ma'idah 5:109).

Each entry: (surah, ayah, kind, word_index, expected_valid).
Run from tajweed-server/ with the tajweed venv:
  python tests/test_reference_sites.py
No model needed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import reference  # noqa: E402

# (surah, ayah, kind, word_index, valid set)
EXPECTED = [
    # 1:1 — بسم الله الرحمن الرحيم
    (1, 1, "madd_normal", 1, {2}),   # الله alif
    (1, 1, "madd_normal", 2, {2}),   # الرحمن alif
    (1, 1, "madd_aared", 3, {2, 4, 6}),  # الرحيم ya (verse-final)
    # 1:2 — الحمد لله رب العالمين
    (1, 2, "madd_normal", 1, {2}),   # لله alif
    (1, 2, "madd_normal", 3, {2}),   # العالمين alif
    (1, 2, "madd_aared", 3, {2, 4, 6}),  # العالمين ya (verse-final)
    # 1:3 — الرحمن الرحيم
    (1, 3, "madd_normal", 0, {2}),   # الرحمن alif
    (1, 3, "madd_aared", 1, {2, 4, 6}),  # الرحيم ya
    # 1:4 — مالك يوم الدين
    (1, 4, "madd_normal", 0, {2}),   # مالك alif
    (1, 4, "madd_aared", 2, {2, 4, 6}),  # الدين ya
    # 1:5 — إياك نعبد وإياك نستعين
    (1, 5, "madd_normal", 0, {2}),   # إياك alif
    (1, 5, "madd_normal", 2, {2}),   # وإياك alif
    (1, 5, "madd_aared", 3, {2, 4, 6}),  # نستعين ya
    # 1:6 — اهدنا الصراط المستقيم
    (1, 6, "madd_normal", 1, {2}),   # الصراط alif
    (1, 6, "madd_aared", 2, {2, 4, 6}),  # المستقيم ya
    # 1:7 — صراط الذين أنعمت عليهم غير المغضوب عليهم ولا الضالين
    (1, 7, "madd_normal", 0, {2}),   # صراط alif
    (1, 7, "madd_normal", 1, {2}),   # الذين ya (mid-verse)
    (1, 7, "madd_normal", 5, {2}),   # المغضوب waw (mid-verse)
    # NOTE: ولا's alif is NOT expected — the phonetizer drops it when
    # followed by الضالين (وَلَا ٱلضَّآلِّينَ -> وَلَ ضضَاااااا...).
    # The model was trained on the same phonetizer output, so it omits
    # it too: no false flag, but the madd is not verified for 1:7.
    (1, 7, "madd_lazim", 8, {6}),    # الضالين alif (6 beats)
    (1, 7, "madd_aared", 8, {2, 4, 6}),  # الضالين ya (verse-final)
    # 78:1 — عم يتساءلون
    (78, 1, "meem_moshaddadah", 0, {1}),  # عم meem shaddah
    (78, 1, "madd_muttasel", 1, {4, 5}),  # يتساءلون alif (4 beats)
    (78, 1, "madd_aared", 1, {2, 4, 6}),  # يتساءلون waw (verse-final)
    # 5:109 — the 9 annotated sites + the munfasel in ماذا
    (5, 109, "madd_normal", 7, {2}),   # قالوا alif + waw (two sites)
    (5, 109, "madd_normal", 7, {2}),
    (5, 109, "madd_normal", 8, {2}),   # لا alif
    (5, 109, "separate_madd", 5, {2, 4, 5}),  # ماذا alif + hamza next
    (5, 109, "separate_madd", 10, {2, 4, 5}),  # لنا alif + إنك
    (5, 109, "noon_moshaddadah", 11, {1}),  # إنك noon shaddah
    (5, 109, "noon_mokhfah", 12, {2}),  # أنت ikhfaa
    (5, 109, "madd_normal", 13, {2}),   # علام alif
    (5, 109, "madd_aared", 14, {2, 4, 6}),  # الغيوب waw
    (5, 109, "qalqalah", 14, {1}),     # الغيوب ب
]


def main():
    by_verse = {}
    for (s, a, kind, wi, valid) in EXPECTED:
        by_verse.setdefault((s, a), []).append((kind, wi, valid))
    failures = 0
    for (s, a), want in sorted(by_verse.items()):
        ref = reference.build_reference(s, a)
        got = [(x["kind"], x["word_index"], set(x["valid"]))
               for x in ref["sites"]]
        print(f"{s}:{a} {ref['uthmani']}")
        for x in ref["sites"]:
            print(f"   {x['kind']:18s} word#{x['word_index']} {x['word']} "
                  f"run={x['run_char']!r} ref_len={x['ref_len']} "
                  f"valid={x['valid']}")
        for (kind, wi, valid) in want:
            if (kind, wi, valid) not in got:
                print(f"  MISSING: {kind} word#{wi} valid={sorted(valid)}")
                failures += 1
        # no unexpected madd_aared / separate / lazim sites
        for (kind, wi, valid) in got:
            if kind in ("madd_aared", "separate_madd", "madd_muttasel",
                        "madd_lazim") and (kind, wi, valid) not in want:
                print(f"  UNEXPECTED: {kind} word#{wi} valid={sorted(valid)}")
                failures += 1
    if failures:
        print(f"FAIL — {failures} site errors")
        sys.exit(1)
    print("PASS — all expected sites found, no unexpected madd sites")


if __name__ == "__main__":
    main()
