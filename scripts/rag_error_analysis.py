import pandas as pd
from pathlib import Path

INPUT_FILE = Path("outputs/rag_mcqa_eval_20260219-174853_detailed.csv")
OUTPUT_FILE = Path("outputs/error_analysis.csv")
MAX_ERRORS = 50

EXPERIMENTS = {
    "baseline": "rag_baseline",
    "llm_only": "llm_only",
    "rerank": "rag_rerank",
    "large": "rag_llm_large",
}


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def load_data(file_path):
    """Load evaluation results and assign question IDs."""
    df = pd.read_csv(file_path)
    df["question_id"] = df.groupby("experiment").cumcount()
    return df


def split_experiments(df):
    """Split dataframe into experiment-specific tables."""
    return {
        name: df[df["experiment"] == exp].copy() for name, exp in EXPERIMENTS.items()
    }


def build_error_table(experiments):
    """Create an error analysis table from baseline mistakes."""
    baseline = experiments["baseline"]

    # Select baseline errors
    errors = baseline.loc[~baseline["is_correct"]].head(MAX_ERRORS).copy()

    print(f"Baseline errors found: {len(errors)}")

    # Correct chunk in top-3
    errors["correct_chunk_top3"] = (
        errors.get("recall@k", 0).gt(0).map({True: "Yes", False: "No"})
    )

    # Merge predictions from other experiments
    result = errors[
        ["question_id", "correct", "predicted", "correct_chunk_top3"]
    ].rename(
        columns={
            "correct": "correct_answer",
            "predicted": "rag_baseline",
        }
    )

    for name in ("llm_only", "rerank", "large"):
        pred_df = experiments[name][["question_id", "predicted"]].rename(
            columns={"predicted": name}
        )
        result = result.merge(pred_df, on="question_id", how="left")

    # Final column rename for readability
    result = result.rename(columns={"large": "rag_32B"})

    return result


def save_output(df, file_path):
    """Save results to CSV."""
    df.to_csv(file_path, index=False)
    print(f"Saved to {file_path}")


# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():
    df = load_data(INPUT_FILE)
    experiments = split_experiments(df)
    error_table = build_error_table(experiments)
    save_output(error_table, OUTPUT_FILE)


if __name__ == "__main__":
    main()
