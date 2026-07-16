import re

# Define the paths for all 10 within-project results (using ASMOTE as the technique)
models = {
    'LiteM': 'results/within_project_ASMOTE_LightGBM.txt',
    'LR': 'results/within_project_ASMOTE_LogisticRegression.txt',
    'RF': 'results/within_project_ASMOTE_RF.txt',
    'XGB': 'results/within_project_ASMOTE_XGB.txt',
    'SVM': 'results/within_project_ASMOTE_SVM.txt',
    'KNN': 'results/within_project_ASMOTE_KNN.txt',
    'ET': 'results/within_project_ASMOTE_ET.txt',
    'ADA': 'results/within_project_ASMOTE_ADA.txt',
    'MLP': 'results/within_project_ASMOTE_MLP.txt',
    'DT': 'results/within_project_ASMOTE_DecisionTree.txt'
}

def readLines(file_name):
    with open(file_name, 'r') as file:
        return file.readlines()

def getProject_items(line: str):
    items = line.split("&")
    project = items.pop(0)
    pattern = re.compile(r'\\enspace|\\%|\\\\|\\n|\\;')  
    items = [float(pattern.sub('', item.strip())) for item in items]
    return project, items

# Read lines from all model result files
model_lines = {name: readLines(path) for name, path in models.items()}

# Get the list of model names in order
model_names_list = list(models.keys())
num_models = len(model_names_list)

new_lines = []
# Assuming all result files have the same number of rows (projects)
num_rows = len(model_lines['LiteM'])

for i in range(num_rows):
    project_name = ""
    project_data = []

    # Parse data for each model for the current row
    for name in model_names_list:
        p_name, datum = getProject_items(model_lines[name][i])
        project_name = p_name  # Keep updating (they should all match)
        project_data.append(datum)

    # Determine maximum values across the models for each column index
    max_values = [max(column) for column in zip(*project_data)]

    # Format the cells, bolding the highest value in each column
    transformed_data = []
    for row in project_data:
        transformed_row = []
        for element, max_value in zip(row, max_values):
            if element == max_value:
                transformed_row.append(f'\\textbf{{{element:.2f}}}')
            else:
                transformed_row.append(f'{element:.2f}')
        transformed_data.append(transformed_row)

    # Build the LaTeX formatted rows for this project group
    formatted_rows = []
    for idx, row in enumerate(transformed_data):
        model_label = model_names_list[idx]
        row_str = " & ".join(row)
        formatted_rows.append(f'& {model_label} & {row_str}\\\\\n')

    # Group the rows under a single multirow header block
    header = f'\\multirow{{{num_models}}}{{*}}{{{project_name.strip()}}}\n'
    divider = '\\cmidrule[0.8pt]{1-22}\n\n' # Adjust index columns to match your LaTeX table width if needed
    
    new_lines = new_lines + [header] + formatted_rows + [divider]

# Write out to the final RQ3 results file
with open('results/RQ3.txt', 'w') as file:
    file.writelines(new_lines)