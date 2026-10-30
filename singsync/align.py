"""Word-level vocal transcription and lyric-line alignment.

``transcribe_words`` wraps faster-whisper (imported lazily, so the rest of
SingSync works without it) and returns word timings. When the user supplies the
real lyrics, transcript tokens are matched against lyric tokens with
``difflib.SequenceMatcher`` so every word carries the index of the lyric line it
belongs to and the last word of each line is flagged ``line_end``.
"""
from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

_PUNCT = re.compile(r"[^\w']+", re.UNICODE)
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "`": "'"})

#: Transcript gap (seconds) that starts a new line when no lyrics are given.
DEFAULT_LINE_GAP_S = 0.65
#: Word count that forces a line break when no lyrics are given.
DEFAULT_LINE_MAX_WORDS = 9


def normalise_word(word: str) -> str:
    """Lower-case, strip accents and punctuation; '' for non-lexical tokens."""
    text = unicodedata.normalize("NFKD", str(word)).translate(_APOSTROPHES)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return _PUNCT.sub("", text.lower()).strip("'")


def tokenise_lyrics(lyrics_text: str) -> list[dict]:
    """Lyric lines -> ``[{"line_idx", "text", "tokens"}]`` (blank lines dropped)."""
    lines = []
    for raw in str(lyrics_text).splitlines():
        text = raw.strip()
        if not text:
            continue
        tokens = [t for t in (normalise_word(w) for w in text.split()) if t]
        if not tokens:
            continue
        lines.append({"line_idx": len(lines), "text": text, "tokens": tokens})
    return lines


def _lines_from_gaps(words: list[dict], gap_s: float, max_words: int) -> None:
    """Assign ``line_idx`` from pauses when no lyrics are available."""
    line_idx = 0
    count = 0
    for i, word in enumerate(words):
        if i:
            gap = float(word["start"]) - float(words[i - 1]["end"])
            if gap >= gap_s or count >= max_words:
                line_idx += 1
                count = 0
        word["line_idx"] = line_idx
        count += 1


def _align_to_lyrics(words: list[dict], lyric_lines: list[dict]) -> None:
    """Map transcript words onto lyric lines via longest-match opcodes."""
    heard = [normalise_word(w["word"]) for w in words]
    flat: list[str] = []
    owner: list[int] = []
    for line in lyric_lines:
        for token in line["tokens"]:
            flat.append(token)
            owner.append(line["line_idx"])

    assigned: list[int | None] = [None] * len(words)
    matcher = SequenceMatcher(None, heard, flat, autojunk=False)
    for i, j, size in matcher.get_matching_blocks():
        for k in range(size):
            assigned[i + k] = owner[j + k]

    # Transcript words whisper heard but the lyrics do not contain (ad-libs,
    # mishearings) inherit the surrounding line; interior runs are split at the
    # midpoint so a word never jumps backwards in line order.
    first = next((v for v in assigned if v is not None), 0)
    for i in range(len(assigned)):
        if assigned[i] is not None:
            break
        assigned[i] = first
    last_known = assigned[0]
    i = 0
    while i < len(assigned):
        if assigned[i] is not None:
            last_known = assigned[i]
            i += 1
            continue
        j = i
        while j < len(assigned) and assigned[j] is None:
            j += 1
        nxt = assigned[j] if j < len(assigned) else last_known
        split = i + (j - i + 1) // 2 if nxt != last_known else j
        for k in range(i, j):
            assigned[k] = last_known if k < split else nxt
        i = j
    for word, line_idx in zip(words, assigned):
        word["line_idx"] = int(line_idx)


def _mark_line_ends(words: list[dict]) -> list[dict]:
    """Flag the last word of every run of equal ``line_idx``; build line list."""
    lines: list[dict] = []
    for i, word in enumerate(words):
        word["line_end"] = i + 1 == len(words) or words[i + 1]["line_idx"] != word["line_idx"]
    start = 0
    for i, word in enumerate(words):
        if not word["line_end"]:
            continue
        chunk = words[start:i + 1]
        lines.append({
            "line_idx": int(word["line_idx"]),
            "text": " ".join(str(w["word"]).strip() for w in chunk).strip(),
            "start": round(float(chunk[0]["start"]), 4),
            "end": round(float(chunk[-1]["end"]), 4),
            "word_from": start,
            "word_to": i + 1,
        })
        start = i + 1
    return lines


def build_alignment(raw_words, lyrics_text: str | None = None, *,
                    line_gap_s: float = DEFAULT_LINE_GAP_S,
                    line_max_words: int = DEFAULT_LINE_MAX_WORDS,
                    meta: dict | None = None) -> dict:
    """Clean raw ``(word, start, end)`` triples into the alignment dict.

    Pure function: this is the part of :func:`transcribe_words` that is testable
    without faster-whisper.
    """
    words: list[dict] = []
    for item in raw_words:
        if isinstance(item, dict):
            text, start, end = item.get("word"), item.get("start"), item.get("end")
        else:
            text, start, end = item
        if text is None or start is None or end is None:
            continue
        text = str(text).strip()
        if not text or not normalise_word(text):
            continue
        start, end = float(start), float(end)
        if end < start:
            start, end = end, start
        if words and start < float(words[-1]["end"]):
            # faster-whisper occasionally overlaps neighbouring words; keep the
            # sequence monotonic so cut boundaries stay well defined.
            start = max(start, float(words[-1]["end"]))
            end = max(end, start)
        words.append({"word": text, "start": round(start, 4), "end": round(end, 4)})

    lyric_lines = tokenise_lyrics(lyrics_text) if lyrics_text else []
    if words:
        if lyric_lines:
            _align_to_lyrics(words, lyric_lines)
        else:
            _lines_from_gaps(words, line_gap_s, line_max_words)
    lines = _mark_line_ends(words)
    if lyric_lines:
        by_idx = {line["line_idx"]: line["text"] for line in lyric_lines}
        for line in lines:
            line["lyric_text"] = by_idx.get(line["line_idx"], "")

    out = {
        "words": words,
        "lines": lines,
        "lyrics_used": bool(lyric_lines),
        "lyric_lines": [{"line_idx": l["line_idx"], "text": l["text"]} for l in lyric_lines],
    }
    if meta:
        out.update(meta)
    return out


def transcribe_words(vocal_path: str | Path, lyrics_text: str | None = None,
                     model: str = "small.en", device: str = "auto",
                     *, compute_type: str | None = None, language: str | None = None,
                     beam_size: int = 5, vad_filter: bool = True) -> dict:
    """Transcribe ``vocal_path`` with word timestamps and align to ``lyrics_text``.

    Requires ``faster-whisper`` (``pip install faster-whisper``). The model is
    loaded on first use and cached per (model, device, compute_type).
    """
    path = Path(vocal_path)
    if not path.exists():
        raise FileNotFoundError("vocal audio not found: %s" % path)
    whisper_model = _load_model(model, device, compute_type)
    segments, info = whisper_model.transcribe(
        str(path), word_timestamps=True, beam_size=beam_size,
        vad_filter=vad_filter, language=language)
    raw: list[dict] = []
    for segment in segments:
        for word in (getattr(segment, "words", None) or []):
            raw.append({"word": word.word, "start": word.start, "end": word.end})
    meta = {
        "source": str(path),
        "model": model,
        "language": getattr(info, "language", language) or "",
        "audio_duration": round(float(getattr(info, "duration", 0.0) or 0.0), 4),
    }
    return build_alignment(raw, lyrics_text, meta=meta)


_MODEL_CACHE: dict[tuple, object] = {}


def _load_model(model: str, device: str, compute_type: str | None):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "faster-whisper is required for transcription. "
            "Install it with: pip install faster-whisper"
        ) from exc
    if device == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"
    key = (model, device, compute_type)
    if key not in _MODEL_CACHE:
        try:
            _MODEL_CACHE[key] = WhisperModel(model, device=device, compute_type=compute_type)
        except Exception:  # pragma: no cover - fall back to the safe combination
            if (model, "cpu", "int8") in _MODEL_CACHE:
                return _MODEL_CACHE[(model, "cpu", "int8")]
            _MODEL_CACHE[key] = WhisperModel(model, device="cpu", compute_type="int8")
    return _MODEL_CACHE[key]


def _cuda_available() -> bool:
    try:  # pragma: no cover - depends on environment
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def load_alignment(json_or_path) -> dict:
    """Load an alignment from a dict, a JSON string or a path to a JSON file."""
    if isinstance(json_or_path, dict):
        data = json_or_path
    else:
        text = str(json_or_path)
        if text.lstrip().startswith("{"):
            data = json.loads(text)
        else:
            data = json.loads(Path(text).read_text(encoding="utf-8"))
    words = data.get("words") or []
    if not isinstance(words, list):
        raise ValueError("alignment has no 'words' list")
    for word in words:
        word.setdefault("line_idx", 0)
        word.setdefault("line_end", False)
    if not data.get("lines"):
        data["lines"] = _mark_line_ends(words)
    return data


def save_alignment(alignment: dict, path: str | Path) -> str:
    """Write an alignment dict to ``path`` as JSON; returns the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(alignment, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out)
