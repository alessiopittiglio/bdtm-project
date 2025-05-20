import pandas as pd
import os
import re

def parse_llm_choice(llm_output_string: str) -> str:
    """
    Extracts the first letter A, B, C, or D from the LLM output.
    Returns the letter or an error string.
    """
    if not llm_output_string:
        return "ERROR_EMPTY_OUTPUT"
    
    # Look for a single letter A, B, C, or D, possibly with spaces around or punctuation
    match = re.match(
        r"^\s*([A-D])(?:[.\s]|$)", llm_output_string.strip(), re.IGNORECASE
    )
    if match:
        return match.group(1).upper()
    else:
        # If no exact match is found, try to find the letter anywhere in the string
        # (this is less restrictive, may give false positives but captures more cases)
        search_match = re.search(r"([A-D])", llm_output_string, re.IGNORECASE)
        if search_match:
            return search_match.group(1).upper()
        return "ERROR_PARSING_CHOICE"


def calculate_accuracy(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate accuracy and other statistics from the detailed results DataFrame.
    Returns a summary DataFrame.
    """
    if results_df is None or results_df.empty:
        return pd.DataFrame()

    valid_predictions_df = results_df[
        results_df['predicted_letter'].isin(["A", "B", "C", "D"])
    ].copy()
    
    valid_predictions_df['is_correct_numeric'] = (
        valid_predictions_df['is_correct'].fillna(False).astype(int)
    )

    summary_list = []
    for config_name, group in valid_predictions_df.groupby('config_name'):
        total_questions_for_config = len(
            results_df[results_df['config_name'] == config_name]
        )
        valid_answers = len(group)
        correct_answers = group['is_correct_numeric'].sum()
        
        accuracy = (correct_answers / valid_answers * 100) if valid_answers > 0 else 0.0
        error_predictions = total_questions_for_config - valid_answers
        
        summary_list.append({
            "config_name": config_name,
            "accuracy_perc": round(accuracy, 2),
            "correct_predictions": correct_answers,
            "valid_predictions_count": valid_answers,
            "error_predictions_count": error_predictions,
            "total_questions_attempted": total_questions_for_config
        })
        
    return pd.DataFrame(summary_list)


def save_results_to_csv(
        detailed_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        results_dir: str,
        base_filename: str
    ):
    """Save detailed and summary DataFrames to CSV files."""
    os.makedirs(results_dir, exist_ok=True)
    
    detailed_filepath = os.path.join(results_dir, f"{base_filename}_detailed.csv")
    summary_filepath = os.path.join(results_dir, f"{base_filename}_summary.csv")

    try:
        detailed_df.to_csv(detailed_filepath, index=False, encoding='utf-8-sig')
        print(f"Detailed results saved to: {detailed_filepath}")
        if summary_df is not None and not summary_df.empty:
            summary_df.to_csv(summary_filepath, index=False, encoding='utf-8-sig')
            print(f"Summary saved to: {summary_filepath}")
        else:
            print("No summary to save.")
    except Exception as e:
        print(f"ERROR while saving CSV results: {e}")
