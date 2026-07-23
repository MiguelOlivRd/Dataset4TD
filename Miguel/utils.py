import os
import re
import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def parse_rq3_table(file_path):
    """Reads a LaTeX table from a file and parses it into a clean Pandas DataFrame."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the table data between \midrule and \bottomrule
    table_body_match = re.search(
        r"\\midrule(.*?)\\bottomrule", content, re.DOTALL
    )
    if not table_body_match:
        raise ValueError(
            "Could not find table body between \\midrule and \\bottomrule."
        )

    table_body = table_body_match.group(1)

    # Clean up LaTeX styling tags
    def clean_cell(text):
        text = text.strip()
        text = re.sub(r"\\textbf\{([^}]+)\}", r"\1", text)  # remove \textbf{}
        text = re.sub(r"\\cellcolor\{[^}]+\}", "", text)  # remove \cellcolor{}
        text = re.sub(
            r"\\multirow\{[^}]+\}\{[^}]+\}", "", text
        )  # remove \multirow{}{}
        text = text.replace("{", "").replace("}", "")  # strip stray braces
        return text.strip()

    parsed_rows = []
    current_project = None

    for line in table_body.split("\n"):
        line = line.strip()
        # Skip comment lines, horizontal lines, or alignment definitions
        if (
            not line
            or line.startswith("%")
            or line.startswith("\\hline")
            or line.startswith("\\cmidrule")
        ):
            continue

        if "&" in line:
            # Strip off trailing LaTeX line breaks
            line_clean = re.sub(r"\\\\.*$", "", line)
            cells = [clean_cell(c) for c in line_clean.split("&")]

            # Ignore the "Average" rows at the bottom of the table
            if any(
                "average" in cell.lower() for cell in cells if isinstance(cell, str)
            ):
                continue

            if len(cells) >= 22:
                project = cells[0]
                technique = cells[1]

                # Propagate project name downward if empty (due to LaTeX multirow)
                if project == "":
                    project = current_project
                else:
                    current_project = project

                # Parse metric columns to floats
                metrics = []
                for val in cells[2:22]:
                    try:
                        metrics.append(float(val))
                    except ValueError:
                        metrics.append(np.nan)

                parsed_rows.append(
                    {
                        "Project": project,
                        "Technique": technique,
                        "Metrics": metrics,
                    }
                )

    # Generate flat multi-level headers: Granularity_Metric
    granularities = ["File", "Class", "Method", "Block"]
    metrics_names = ["P", "R", "F1", "AUC", "MCC"]
    columns = [f"{g}_{m}" for g in granularities for m in metrics_names]

    df_list = []
    for row in parsed_rows:
        row_dict = {"Project": row["Project"], "Technique": row["Technique"]}
        for col_name, val in zip(columns, row["Metrics"]):
            row_dict[col_name] = val
        df_list.append(row_dict)

    df = pd.DataFrame(df_list)
    return df.iloc[:-3]  # Deletes the last 3 rows


def clean_cell(text):
    """Removes LaTeX commands, backslashes, percent signs, and whitespace."""
    # Remove LaTeX commands like \textbf{...} but keep the inner text
    text = re.sub(r"\\textbf\{([^}]+)\}", r"\1", text)
    # Remove other LaTeX commands like \enspace, \\
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    # Remove leftover backslashes, percent signs, and extra whitespace
    text = text.replace("\\", "").replace("%", "").strip()
    return text


def parse_within_project_file(filepath):
    """Parses a single file and extracts F1-scores for all granularities."""
    data = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "&" not in line:
                continue

            # Split the row by the LaTeX column separator
            parts = [clean_cell(p) for p in line.split("&")]

            # Ensure we have a valid data row (1 project name + 20 metrics)
            if len(parts) < 21:
                continue

            project_name = parts[0]

            # F1-scores are at indices 2 (File), 7 (Class), 12 (Method), and 17 (Block)
            try:
                f1_scores = {
                    "Project": project_name,
                    "File_F1": float(parts[2]),
                    "Class_F1": float(parts[7]),
                    "Method_F1": float(parts[12]),
                    "Block_F1": float(parts[17]),
                }
                data.append(f1_scores)
            except ValueError:
                # Skip rows where parsing to float fails (e.g., table headers)
                continue

    return pd.DataFrame(data)


def read_data(GRANULARITIES, DF_FOLDER_PATH, PROJECTS):

    def find_file(folder_path, prefix):
        # folder.glob('prefix*') looks for anything starting with the prefix
        folder = Path(folder_path)
        return next(folder.glob(f"{prefix}*"))

    # granularity -> project -> dataframe
    data = {}

    for granularity in tqdm(GRANULARITIES):
        data[granularity] = {}
        

        for project in PROJECTS:
            
            all_projects_folder_path = f"{DF_FOLDER_PATH}/{granularity}/"
            project_file = find_file(folder_path=all_projects_folder_path, prefix=project.lower())
            
            project_df = pd.read_csv(project_file)
            project_name = project.split("-")[0]
            data[granularity][project_name] = project_df

    return data