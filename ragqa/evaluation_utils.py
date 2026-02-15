import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

VALID_CHOICES = {"A", "B", "C", "D"}

CHOICE_PATTERN = re.compile(r"^\s*([A-D])(?:[.\s]|$)", re.IGNORECASE)

CHOICE_FALLBACK_PATTERN = re.compile(r"\b([A-D])\b", re.IGNORECASE)


def parse_llm_choice(output: str) -> str:
    """Extract the first valid choice (A, B, C, or D) from LLM output."""
    if not output:
        return None

    text = output.strip()

    match = CHOICE_PATTERN.match(text)
    if match:
        return match.group(1).upper()

    fallback = CHOICE_FALLBACK_PATTERN.search(text)
    if fallback:
        return fallback.group(1).upper()

    return None


def calculate_accuracy(results: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy and summary statistics per experiment."""
    if results is None or results.empty:
        return pd.DataFrame()

    df = results.copy()

    df["is_valid"] = df["correct"].isin(VALID_CHOICES)
    df["is_correct_numeric"] = df["is_correct"].fillna(False).astype(int)

    grouped = df.groupby("experiment", sort=False)

    summary = grouped.agg(
        total_questions=("experiment", "size"),
        valid_predictions=("is_valid", "sum"),
        correct_predictions=("is_correct_numeric", "sum"),
    ).reset_index()

    summary["error_predictions"] = (
        summary["total_questions"] - summary["valid_predictions"]
    )

    summary["accuracy_percent"] = (
        (
            100
            * summary["correct_predictions"]
            / summary["valid_predictions"].replace(0, pd.NA)
        )
        .fillna(0.0)
        .round(2)
    )

    return summary[
        [
            "experiment",
            "accuracy_percent",
            "correct_predictions",
            "valid_predictions",
            "error_predictions",
            "total_questions",
        ]
    ]


def save_results(
    detailed: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str,
    base_name: str,
):
    """Save detailed and summary results as CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    detailed_file = output_path / f"{base_name}_detailed.csv"
    summary_file = output_path / f"{base_name}_summary.csv"

    try:
        detailed.to_csv(detailed_file, index=False, encoding="utf-8-sig")
        logger.info("Detailed results saved to %s", detailed_file)

        if summary is not None and not summary.empty:
            summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
            logger.info("Summary results saved to %s", summary_file)
        else:
            logger.info("Summary is empty. Skipping save.")

    except OSError:
        logger.exception("Failed to save results to %s", output_path)
