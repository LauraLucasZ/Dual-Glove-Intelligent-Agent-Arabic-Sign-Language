"""
Evaluate pre-generated Edge TTS audio used by the Arabic Sign Language glove
words voice pipeline. Does not regenerate or modify audio files.
"""

from __future__ import annotations

import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration (edit paths or add manual gesture -> audio overrides here)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "arabic_gestures_corrected.csv"
LABEL_ENCODER_PATH = BASE_DIR / "ArabicGloves_LabelEncoder.pkl"
WORDS_CHOSEN_DIR = BASE_DIR / "generated_audio_words" / "edge_tts_words" / "chosen"
RESULTS_PATH = BASE_DIR / "voice_output_evaluation_results.txt"

# Optional manual overrides: gesture label -> absolute or project-relative MP3 path.
MANUAL_GESTURE_AUDIO_MAP: dict[str, str | Path] = {
    # "gesture_label": "generated_audio_words/edge_tts_words/chosen/word_example_v01.mp3",
}

# Spelling/dialect aliases (same as words_voice_prediction_pipeline.ipynb).
WORD_ALIASES = {
    "عاوز": "عايز",
    "بحب": "بحبك",
    "شكراً": "شكرا",
    "عفواً": "عفوا",
}


@dataclass
class EvaluationResults:
    gesture_labels: list[str] = field(default_factory=list)
    existing_audio: list[str] = field(default_factory=list)
    missing_audio: list[str] = field(default_factory=list)
    loaded_audio: list[str] = field(default_factory=list)
    failed_audio: list[tuple[str, str]] = field(default_factory=list)
    loading_times: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_labels(self) -> int:
        return len(self.gesture_labels)

    @property
    def coverage_percent(self) -> float:
        if self.total_labels == 0:
            return 0.0
        return (len(self.existing_audio) / self.total_labels) * 100.0

    @property
    def average_loading_time(self) -> float:
        if not self.loading_times:
            return 0.0
        return sum(self.loading_times) / len(self.loading_times)

    @property
    def passed(self) -> bool:
        return (
            self.total_labels > 0
            and not self.missing_audio
            and not self.failed_audio
        )


def normalize_label(value: str) -> str:
    """Normalize Arabic/Latin labels to match dataset labels and cached filenames."""
    text = str(value).strip().lower()
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )
    replacements = {
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0622": "\u0627",
        "\u0629": "\u0647",
        "\u0649": "\u064a",
        "\u0626": "\u064a",
        "\u0624": "\u0648",
    }
    for src, dest in replacements.items():
        text = text.replace(src, dest)
    text = re.sub(r"[^\w\u0600-\u06FF]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def word_label_from_filename(path: Path) -> str:
    stem = path.stem
    if stem.startswith("word_"):
        stem = stem[len("word_") :]
    stem = re.sub(r"_v\d+.*$", "", stem)
    return stem.replace("_", " ")


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def load_gesture_labels() -> list[str]:
    """Load gesture labels from the saved label encoder, with CSV fallback."""
    if LABEL_ENCODER_PATH.exists():
        try:
            import joblib

            label_encoder = joblib.load(LABEL_ENCODER_PATH)
            classes = getattr(label_encoder, "classes_", None)
            if classes is not None and len(classes) > 0:
                return [str(label).strip() for label in classes]
        except Exception as exc:
            print(f"Warning: could not load label encoder ({exc}); falling back to CSV.")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing gesture label sources: {LABEL_ENCODER_PATH.name} and {DATASET_PATH.name}"
        )

    import pandas as pd

    df = pd.read_csv(DATASET_PATH)
    if "label" not in df.columns:
        raise ValueError(f"Dataset missing 'label' column: {DATASET_PATH}")
    labels = sorted(df["label"].dropna().astype(str).str.strip().unique().tolist())
    if not labels:
        raise ValueError(f"No gesture labels found in {DATASET_PATH}")
    return labels


def build_word_audio_lookup() -> dict[str, Path]:
    """
    Build gesture-to-audio lookup from approved MP3 files (same logic as the notebook).
    Manual overrides in MANUAL_GESTURE_AUDIO_MAP are applied last.
    """
    lookup: dict[str, Path] = {}

    if WORDS_CHOSEN_DIR.exists():
        for audio_path in sorted(WORDS_CHOSEN_DIR.glob("*.mp3")):
            label = word_label_from_filename(audio_path)
            keys = {label, label.replace(" ", "_"), audio_path.stem}
            for key in keys:
                lookup[normalize_label(key)] = audio_path.resolve()
    else:
        print(f"Warning: words audio folder not found: {WORDS_CHOSEN_DIR}")

    for alias, canonical in WORD_ALIASES.items():
        canonical_path = lookup.get(normalize_label(canonical))
        if canonical_path is not None:
            lookup.setdefault(normalize_label(alias), canonical_path)

    for gesture_label, audio_ref in MANUAL_GESTURE_AUDIO_MAP.items():
        lookup[normalize_label(gesture_label)] = resolve_path(audio_ref)

    return lookup


def resolve_audio_path(gesture_label: str, lookup: dict[str, Path]) -> Path | None:
    return lookup.get(normalize_label(gesture_label))


def load_audio_for_playback(audio_path: Path) -> tuple[bool, str, float]:
    """
    Load audio and measure preparation time (decode to playable segment).
    Returns (success, error_message, elapsed_seconds).
    """
    start = time.perf_counter()
    try:
        from pydub import AudioSegment

        segment = AudioSegment.from_file(audio_path)
        if len(segment) <= 0:
            elapsed = time.perf_counter() - start
            return False, "Audio file decoded to zero duration.", elapsed
        elapsed = time.perf_counter() - start
        return True, "", elapsed
    except ImportError:
        pass
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return False, f"pydub load failed: {exc}", elapsed

    try:
        from mutagen.mp3 import MP3

        info = MP3(audio_path)
        if info.info is None or info.info.length <= 0:
            elapsed = time.perf_counter() - start
            return False, "MP3 metadata reports zero duration.", elapsed
        elapsed = time.perf_counter() - start
        return True, "", elapsed
    except ImportError:
        pass
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return False, f"mutagen load failed: {exc}", elapsed

    try:
        data = audio_path.read_bytes()
        if len(data) < 128:
            elapsed = time.perf_counter() - start
            return False, "File is too small to be a valid MP3.", elapsed
        if not (data.startswith(b"ID3") or data[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}):
            elapsed = time.perf_counter() - start
            return False, "File does not look like a valid MP3.", elapsed
        elapsed = time.perf_counter() - start
        return True, "", elapsed
    except OSError as exc:
        elapsed = time.perf_counter() - start
        return False, f"Could not read file: {exc}", elapsed


def evaluate_voice_output() -> EvaluationResults:
    results = EvaluationResults()
    results.gesture_labels = load_gesture_labels()
    lookup = build_word_audio_lookup()

    if not lookup:
        results.warnings.append(
            "No gesture-to-audio mappings were discovered. "
            "Check WORDS_CHOSEN_DIR or add entries to MANUAL_GESTURE_AUDIO_MAP."
        )

    for gesture_label in results.gesture_labels:
        audio_path = resolve_audio_path(gesture_label, lookup)
        if audio_path is None:
            results.missing_audio.append(gesture_label)
            continue

        if not audio_path.exists():
            results.missing_audio.append(gesture_label)
            results.warnings.append(
                f"Mapped audio path does not exist for '{gesture_label}': {audio_path}"
            )
            continue

        results.existing_audio.append(gesture_label)
        ok, error, elapsed = load_audio_for_playback(audio_path)
        if ok:
            results.loaded_audio.append(gesture_label)
            results.loading_times.append(elapsed)
        else:
            results.failed_audio.append((gesture_label, error))

    return results


def format_summary(results: EvaluationResults) -> str:
    lines = [
        "Voice Output Evaluation Summary",
        "================================",
        f"Total gesture labels: {results.total_labels}",
        f"Existing audio files: {len(results.existing_audio)}",
        f"Missing audio files: {len(results.missing_audio)}",
        f"Successfully loaded audio files: {len(results.loaded_audio)}",
        f"Failed audio files: {len(results.failed_audio)}",
        f"Audio coverage percentage: {results.coverage_percent:.2f}%",
        f"Average loading time (seconds): {results.average_loading_time:.4f}",
        f"Final status: {'Passed' if results.passed else 'Failed'}",
    ]

    if results.missing_audio:
        lines.append("")
        lines.append("Missing audio for gesture labels:")
        for label in results.missing_audio:
            lines.append(f"  - {label}")

    if results.failed_audio:
        lines.append("")
        lines.append("Failed to load audio for gesture labels:")
        for label, error in results.failed_audio:
            lines.append(f"  - {label}: {error}")

    if results.warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in results.warnings:
            lines.append(f"  - {warning}")

    lines.append("")
    lines.append(f"Results saved to: {RESULTS_PATH.resolve()}")
    return "\n".join(lines)


def main() -> int:
    try:
        results = evaluate_voice_output()
    except Exception as exc:
        message = f"Voice output evaluation failed before completion: {exc}"
        print(message, file=sys.stderr)
        RESULTS_PATH.write_text(message + "\n", encoding="utf-8")
        return 1

    summary = format_summary(results)
    print(summary)
    RESULTS_PATH.write_text(summary + "\n", encoding="utf-8")
    return 0 if results.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
