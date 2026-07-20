import re
import pandas as pd

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