"""Build the reference QPS for any verse and find every tajweed site in it.

The key insight from the Nov 12 probe: the reference QPS ALREADY encodes
every rule. The phonetizer writes:
  * madd letters as runs (ا / ۥ / ۦ repeated per beat: normal 2,
    mottasel 4, lazim 6, aared 2 in the reference);
  * shaddah as a doubled consonant (لل, رر, نننن for a ghunnah noon);
  * ikhfaa noon as ں (vs plain ن for izhar);
  * qalqalah as the ڇ marker after the qalqalah letter.
So site-finding is reading what the phonetizer wrote — no hand list of
ikhfaa letters, no hardcoded verse anchors (the old qdat_parse.py only
worked for 5:109; this works for any verse).

Madd classification (per run, in order):
  1. aared  — the run is in the LAST word of the verse and after it there
     is exactly one consonant (+ optional short vowel / qalqalah marker)
     before the word ends. That is the madd before the letter you stop
     on at waqf. Valid {2,4,6} — the PERMANENT waqf rule (Sory, Nov 12).
     (A run at the very end of the last word, with nothing after it, is
     NOT aared — there is no stopped-on letter after it — it is normal.)
  2. munfasel (separate) — the run ends its word and the NEXT word starts
     with hamza (ء). Valid {2,4,5}.
  3. mottasel — the run is mid-word and the next char in the word is
     hamza (ء). Valid {4,5}.
  4. lazim — the run is mid-word and the next char is a doubled
     consonant (shaddah). Valid {6}.
  5. normal (tabi'i) — everything else. Valid {2}.

Ghunnah: a ن run or م run of length >= 2 inside a word is a shaddah
noon/meem. Ikhfaa: any ں run. Qalqalah: any ڇ marker (the letter just
before it is the qalqalah letter).
"""
import re

from quran_transcript import Aya, MoshafAttributes, quran_phonetizer

from . import config

MADD_LETTERS = {"ا", "ۥ", "ۦ"}
DIACRITICS = {"َ", "ُ", "ِ", "ً", "ٌ", "ٍ", "ْ", "ّ", "ٰ"}
HAMZA = "ء"
QALQALAH_MARK = "ڇ"
IKHFAA_NOON = "ں"
GHUNNAH_LETTERS = {"ن", "م"}

LABELS = {
    "madd_normal": "Madd tabi'i (natural lengthening)",
    "separate_madd": "Madd munfasel (separate lengthening)",
    "madd_muttasel": "Madd muttasel (connected lengthening)",
    "madd_lazim": "Madd lazim (compulsory lengthening)",
    "madd_aared": "Madd aarid (stopping lengthening)",
    "noon_moshaddadah": "Ghunnah on doubled noon",
    "meem_moshaddadah": "Ghunnah on doubled meem",
    "noon_mokhfah": "Ikhfaa (hidden noon)",
    "qalqalah": "Qalqalah (echoing letter)",
}

_MOSHAF = MoshafAttributes(**config.MOSHAF)


def _runs(word, chars):
    """Maximal runs of chars in word: list of (start, end, char)."""
    out = []
    i = 0
    while i < len(word):
        if word[i] in chars:
            j = i + 1
            while j < len(word) and word[j] == word[i]:
                j += 1
            out.append((i, j, word[i]))
            i = j
        else:
            i += 1
    return out


def _tail_is_single_consonant(word, end):
    """True if word[end:] is exactly one consonant (+ optional short
    vowel / qalqalah marker) — i.e. the run sits in the final syllable
    of the word, before the letter you stop on."""
    tail = word[end:].replace(QALQALAH_MARK, "")
    if len(tail) == 1:
        return tail[0] not in DIACRITICS and tail[0] not in MADD_LETTERS
    if len(tail) == 2:
        return (tail[0] not in DIACRITICS and tail[0] not in MADD_LETTERS
                and tail[1] in DIACRITICS)
    return False


def find_sites(qps_words, uthmani_words):
    """All tajweed sites in a verse, in reading order.

    qps_words: the reference QPS split on spaces.
    uthmani_words: the Uthmani text split on spaces (display only).
    Returns a list of site dicts.
    """
    n = len(qps_words)
    sites = []
    # no-space offsets per word, for the CTC alignment target
    offsets = []
    pos = 0
    for w in qps_words:
        offsets.append(pos)
        pos += len(w)

    for wi, w in enumerate(qps_words):
        display = (uthmani_words[wi] if wi < len(uthmani_words) else w)
        last_word = wi == n - 1
        next_word = qps_words[wi + 1] if wi + 1 < n else ""

        # --- madd runs ---
        for (s, e, ch) in _runs(w, MADD_LETTERS):
            after = w[e] if e < len(w) else None
            word_final = e == len(w)
            if last_word and _tail_is_single_consonant(w, e):
                kind = "madd_aared"
            elif word_final and next_word.startswith(HAMZA):
                kind = "separate_madd"
            elif not word_final and after == HAMZA:
                kind = "madd_muttasel"
            elif (not word_final and e + 1 < len(w)
                    and w[e] == w[e + 1]):
                kind = "madd_lazim"
            else:
                kind = "madd_normal"
            sites.append(_site(kind, display, w, s, e, ch,
                               offsets[wi], wi))

        # --- ghunnah: doubled noon / meem inside the word ---
        for (s, e, ch) in _runs(w, GHUNNAH_LETTERS):
            if e - s >= 2:
                kind = ("noon_moshaddadah" if ch == "ن"
                        else "meem_moshaddadah")
                sites.append(_site(kind, display, w, s, e, ch,
                                   offsets[wi], wi))

        # --- ikhfaa: any ں run ---
        for (s, e, ch) in _runs(w, {IKHFAA_NOON}):
            sites.append(_site("noon_mokhfah", display, w, s, e, ch,
                               offsets[wi], wi))

        # --- qalqalah: any ڇ marker (letter just before it is the site) ---
        for m in re.finditer(QALQALAH_MARK, w):
            j = m.start()
            if j == 0:
                continue
            sites.append(_site("qalqalah", display, w, j - 1, j, w[j - 1],
                               offsets[wi], wi, qalqalah_pos=j))

    # reading order = word order, then position in the word
    sites.sort(key=lambda s: (s["word_index"], s["start_in_word"]))
    for i, s in enumerate(sites):
        s["id"] = f"site_{i + 1}"
    return sites


def _site(kind, display, word, s, e, ch, word_offset, wi, qalqalah_pos=None):
    """One site. `anchor` = the 2 chars before the run in the no-space
    reference (used for the sequential search in the predicted QPS)."""
    nospace_off = word_offset + s
    # anchor needs the full no-space string; it is filled in by
    # build_reference() after the sites are found.
    return {
        "kind": kind,
        "rule": kind,
        "label": LABELS[kind],
        "valid": sorted(config.VALID[kind]),
        "word": display,
        "word_index": wi,
        "start_in_word": s,
        "end_in_word": e,
        "run_char": ch,
        "ref_len": e - s if kind != "qalqalah" else 1,
        "nospace_offset": nospace_off,
        "qalqalah_pos": qalqalah_pos,
        "anchor": None,  # filled in by build_reference()
    }


def build_reference(surah, ayah):
    """Reference QPS + sites for one verse.

    Returns {surah, ayah, uthmani, qps_spaced, qps_nospace, sites}.
    Raises ValueError on a bad surah/ayah.
    """
    try:
        uthmani = Aya(surah, ayah).get().uthmani
        qps_spaced = quran_phonetizer(uthmani, _MOSHAF).phonemes
        qps_nospace = quran_phonetizer(uthmani, _MOSHAF,
                                       remove_spaces=True).phonemes
    except Exception as exc:
        raise ValueError(f"no such verse: {surah}:{ayah}") from exc
    qps_words = qps_spaced.split()
    uthmani_words = uthmani.split()
    sites = find_sites(qps_words, uthmani_words)
    for s in sites:
        off = s["nospace_offset"]
        s["anchor"] = qps_nospace[max(0, off - 2):off]
    return {
        "surah": surah,
        "ayah": ayah,
        "uthmani": uthmani,
        "qps_spaced": qps_spaced,
        "qps_nospace": qps_nospace,
        "sites": sites,
    }
