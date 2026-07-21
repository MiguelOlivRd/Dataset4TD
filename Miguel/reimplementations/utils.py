# Data reading
import pandas as pd

from pathlib import Path
from tqdm import tqdm

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