"""
Evaluate pre-generated Edge TTS audio used by the Arabic Sign Language glove project.

Mapping source: same approved letter/number tables as full_voice_prediction_pipeline.ipynb
(no standalone JSON mapping file exists in this project).

Gesture labels are loaded from label_encoder.pkl when available.
"""

from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
GENERATED_AUDIO_DIR = BASE_DIR / "generated_audio"
LETTERS_CHOSEN_DIR = GENERATED_AUDIO_DIR / "edge_tts_mapping_v2" / "chosen"
NUMBERS_CHOSEN_DIR = GENERATED_AUDIO_DIR / "edge_tts_numbers_mapping_v1" / "chosen"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
RESULTS_PATH = BASE_DIR / "voice_output_evaluation_results.txt"

# Optional override: set to a list of gesture labels to evaluate instead of label_encoder.pkl
MANUAL_GESTURE_LABELS: list[str] | None = None

# Optional override: gesture label -> relative audio path (from BASE_DIR)
MANUAL_GESTURE_AUDIO_MAP: dict[str, str] = {}

# Approved cached MP3 filenames (mirrors full_voice_prediction_pipeline.ipynb)
APPROVED_LETTER_FILES = [
    {"label": "ا", "approved_filename": "letter_ا_alif.mp3"},
    {"label": "ب", "approved_filename": "letter_ب_cut_0.45.mp3"},
    {"label": "ت", "approved_filename": "letter_ت_teh.mp3"},
    {"label": "ث", "approved_filename": "comp_seh_ث_ث_ه.mp3"},
    {"label": "ج", "approved_filename": "letter_ج_geem.mp3"},
    {"label": "ح", "approved_filename": "letter_ح_ha.mp3"},
    {"label": "خ", "approved_filename": "letter_خ_kha.mp3"},
    {"label": "د", "approved_filename": "comp_dal_د_دال_bang.mp3"},
    {"label": "ذ", "approved_filename": "comp_zal_ذ_ذال_bang.mp3"},
    {"label": "ر", "approved_filename": "letter_ر_ra.mp3"},
    {"label": "ز", "approved_filename": "comp_zay_ز_zeen_latin_long.mp3"},
    {"label": "س", "approved_filename": "comp_seen_س_seen_split_cut.mp3"},
    {"label": "ش", "approved_filename": "letter_ش_sheen.mp3"},
    {"label": "ص", "approved_filename": "comp_sad_ص_saad_long_sukoon.mp3"},
    {"label": "ض", "approved_filename": "comp_dad_ض_daad_long_sukoon.mp3"},
    {"label": "ط", "approved_filename": "comp_ta_ط_ط_ه.mp3"},
    {"label": "ظ", "approved_filename": "letter_ظ_dha.mp3"},
    {"label": "ع", "approved_filename": "letter_ع_ein.mp3"},
    {"label": "غ", "approved_filename": "letter_غ_ghein.mp3"},
    {"label": "ف", "approved_filename": "comp_faa_ف_ف_ه.mp3"},
    {"label": "ق", "approved_filename": "comp_qaf_ق_qaf_split_cut_v2.mp3"},
    {"label": "ك", "approved_filename": "comp_kaf_ك_كاف_sukoon.mp3"},
    {"label": "ل", "approved_filename": "comp_lam_ل_lam_latin.mp3"},
    {"label": "م", "approved_filename": "letter_م_meem.mp3"},
    {"label": "ن", "approved_filename": "letter_ن_noon.mp3"},
    {"label": "ه", "approved_filename": "comp_ha_ه_ه_ه.mp3"},
    {"label": "و", "approved_filename": "comp_waw_و_واو_diac.mp3"},
    {"label": "ي", "approved_filename": "letter_ي_yeh.mp3"},
]

APPROVED_NUMBER_FILES = [
    {"label": "0", "approved_filename": "num_00_v02.mp3"},
    {"label": "1", "approved_filename": "num_01.mp3"},
    {"label": "2", "approved_filename": "num_02_atneen_bang_cut_0_59.mp3"},
    {"label": "3", "approved_filename": "num_03.mp3"},
    {"label": "4", "approved_filename": "num_04.mp3"},
    {"label": "5", "approved_filename": "num_05.mp3"},
    {"label": "6", "approved_filename": "num_06.mp3"},
    {"label": "7", "approved_filename": "num_07.mp3"},
    {"label": "8", "approved_filename": "num_08.mp3"},
    {"label": "9", "approved_filename": "num_09.mp3"},
    {"label": "10", "approved_filename": "num_10_asharah_sukoon.mp3"},
]

LETTER_LABEL_ALIASES = {"أ": "ا", "آ": "ا", "إ": "ا", "ٱ": "ا"}

NUMBER_WORD_TO_DIGIT = {
    "صفر": "0",
    "واحد": "1",
    "اثنين": "2",
    "ثلاثة": "3",
    "أربعة": "4",
    "خمسة": "5",
    "ستة": "6",
    "سبعة": "7",
    "ثمانية": "8",
    "تسعة": "9",
    "عشرة": "10",
}


@dataclass
class LabelResult:
    label: str
    relative_path: str | None = None
    absolute_path: Path | None = None
    exists: bool = False
    loaded: bool = False
    load_time_s: float | None = None
    error: str | None = None


@dataclass
class EvaluationSummary:
    total_gesture_labels: int = 0
    existing_audio_files: int = 0
    missing_audio_files: int = 0
    successfully_loaded: int = 0
    failed_audio_files: int = 0
    average_loading_time_s: float = 0.0
    coverage_percent: float = 0.0
    final_status: str = "Failed"
    missing_details: list[str] = field(default_factory=list)
    failed_details: list[str] = field(default_factory=list)


def _rel_chosen_path(chosen_dir: Path, filename: str) -> str:
    return (chosen_dir / filename).relative_to(BASE_DIR).as_posix()


def build_project_audio_maps() -> tuple[dict[str, str], dict[str, str]]:
    letter_map = {
        row["label"]: _rel_chosen_path(LETTERS_CHOSEN_DIR, row["approved_filename"])
        for row in APPROVED_LETTER_FILES
    }
    number_map = {
        row["label"]: _rel_chosen_path(NUMBERS_CHOSEN_DIR, row["approved_filename"])
        for row in APPROVED_NUMBER_FILES
    }
    return letter_map, number_map


def resolve_audio_path(
    label: str,
    letter_map: dict[str, str],
    number_map: dict[str, str],
) -> str | None:
    if label in MANUAL_GESTURE_AUDIO_MAP:
        return MANUAL_GESTURE_AUDIO_MAP[label]

    tok = str(label).strip()
    if not tok:
        return None

    letter_key = LETTER_LABEL_ALIASES.get(tok, tok)
    if letter_key in letter_map:
        return letter_map[letter_key]

    if tok in number_map:
        return number_map[tok]

    digit_key = NUMBER_WORD_TO_DIGIT.get(tok)
    if digit_key and digit_key in number_map:
        return number_map[digit_key]

    return None


def load_gesture_labels() -> list[str]:
    if MANUAL_GESTURE_LABELS is not None:
        return list(MANUAL_GESTURE_LABELS)

    if not LABEL_ENCODER_PATH.is_file():
        raise FileNotFoundError(
            f"label_encoder.pkl not found at {LABEL_ENCODER_PATH}. "
            "Set MANUAL_GESTURE_LABELS at the top of this file."
        )

    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "joblib is required to load gesture labels from label_encoder.pkl."
        ) from exc

    encoder = joblib.load(LABEL_ENCODER_PATH)
    return [str(label) for label in encoder.classes_]


def load_audio_for_playback(path: Path) -> tuple[object, int]:
    """Load audio into memory (same preparation step needed before playback)."""
    try:
        import librosa

        audio, sample_rate = librosa.load(str(path), sr=None, mono=True)
        return audio, int(sample_rate)
    except ImportError:
        pass

    try:
        import soundfile as sf

        audio, sample_rate = sf.read(str(path), always_2d=False)
        return audio, int(sample_rate)
    except ImportError as exc:
        raise ImportError(
            "Install librosa or soundfile to validate MP3 loading."
        ) from exc


def evaluate_label(
    label: str,
    letter_map: dict[str, str],
    number_map: dict[str, str],
) -> LabelResult:
    result = LabelResult(label=label)
    relative_path = resolve_audio_path(label, letter_map, number_map)
    result.relative_path = relative_path

    if relative_path is None:
        result.error = "No audio mapping found for gesture label"
        return result

    absolute_path = BASE_DIR / relative_path
    result.absolute_path = absolute_path
    result.exists = absolute_path.is_file()

    if not result.exists:
        result.error = "Mapped audio file path does not exist"
        return result

    try:
        start = time.perf_counter()
        audio, sample_rate = load_audio_for_playback(absolute_path)
        elapsed = time.perf_counter() - start

        if getattr(audio, "size", None) == 0:
            result.error = "Audio file loaded but contains no samples"
            return result
        if not hasattr(audio, "size") and hasattr(audio, "__len__") and len(audio) == 0:
            result.error = "Audio file loaded but contains no samples"
            return result

        if sample_rate <= 0:
            result.error = f"Invalid sample rate: {sample_rate}"
            return result

        result.loaded = True
        result.load_time_s = elapsed
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"

    return result


def run_evaluation() -> tuple[EvaluationSummary, list[LabelResult]]:
    letter_map, number_map = build_project_audio_maps()
    gesture_labels = load_gesture_labels()
    results: list[LabelResult] = []

    for label in gesture_labels:
        results.append(evaluate_label(label, letter_map, number_map))

    load_times = [r.load_time_s for r in results if r.load_time_s is not None]
    missing = [r for r in results if r.relative_path is None or not r.exists]
    failed = [r for r in results if r.exists and not r.loaded]

    summary = EvaluationSummary(
        total_gesture_labels=len(results),
        existing_audio_files=sum(1 for r in results if r.exists),
        missing_audio_files=len(missing),
        successfully_loaded=sum(1 for r in results if r.loaded),
        failed_audio_files=len(failed),
        average_loading_time_s=(sum(load_times) / len(load_times)) if load_times else 0.0,
    )

    if summary.total_gesture_labels > 0:
        coverage = (summary.existing_audio_files / summary.total_gesture_labels) * 100.0
    else:
        coverage = 0.0

    summary.coverage_percent = coverage

    summary.missing_details = [
        f"{r.label}: {r.error or 'missing'}"
        + (f" ({r.relative_path})" if r.relative_path else "")
        for r in missing
    ]
    summary.failed_details = [
        f"{r.label}: {r.error} ({r.relative_path})" for r in failed
    ]

    if summary.missing_audio_files == 0 and summary.failed_audio_files == 0:
        summary.final_status = "Passed"
    else:
        summary.final_status = "Failed"

    return summary, results


def format_report(summary: EvaluationSummary) -> str:
    lines = [
        "Voice Output Evaluation Results",
        "===============================",
        f"Project directory: {BASE_DIR}",
        f"Mapping source: full_voice_prediction_pipeline.ipynb (approved chosen MP3s)",
        "",
        f"Total gesture labels: {summary.total_gesture_labels}",
        f"Existing audio files: {summary.existing_audio_files}",
        f"Missing audio files: {summary.missing_audio_files}",
        f"Successfully loaded audio files: {summary.successfully_loaded}",
        f"Failed audio files: {summary.failed_audio_files}",
        f"Audio coverage percentage: {summary.coverage_percent:.1f}%",
        f"Average loading time (seconds): {summary.average_loading_time_s:.4f}",
        f"Final status: {summary.final_status}",
    ]

    if summary.missing_details:
        lines.extend(["", "Missing audio details:"])
        lines.extend(f"  - {item}" for item in summary.missing_details)

    if summary.failed_details:
        lines.extend(["", "Failed load details:"])
        lines.extend(f"  - {item}" for item in summary.failed_details)

    return "\n".join(lines) + "\n"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    try:
        summary, _ = run_evaluation()
        report = format_report(summary)
        print(report, end="")
        RESULTS_PATH.write_text(report, encoding="utf-8")
        print(f"\nResults saved to: {RESULTS_PATH.resolve()}")
        return 0 if summary.final_status == "Passed" else 1
    except Exception as exc:
        error_report = (
            "Voice Output Evaluation Results\n"
            "===============================\n"
            f"Evaluation aborted: {type(exc).__name__}: {exc}\n\n"
            f"{traceback.format_exc()}"
            f"Final status: Failed\n"
        )
        print(error_report, end="", file=sys.stderr)
        try:
            RESULTS_PATH.write_text(error_report, encoding="utf-8")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
