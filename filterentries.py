import pandas as pd
import os
import numpy as np

current_directory = os.path.dirname(__file__)
current_csv = pd.read_csv(current_directory+"/../dataset.csv")
all_ids = current_csv["id"].values
all_folders = os.listdir(current_directory+"/Dataset/images")

filter = np.isin(all_ids, all_folders)

new_csv = current_csv[filter]
new_csv.to_csv(current_directory+"/dataset.csv", index=False)