import pandas as pd
import os
import numpy as np

current_directory = os.path.dirname(__file__)
current_csv = pd.read_csv(current_directory+"/../dataset.csv")
all_ids = current_csv["id"].values
all_folders = os.listdir(current_directory+"/Dataset/images")

filter = np.isin(all_ids, all_folders)

new_csv = current_csv[filter]

name_number_dict = {}
check_num_images_table = new_csv["num_images"].values
real_count = [len(os.listdir(current_directory+"/Dataset/images/"+str(id))) for id in new_csv["id"].values]

filter2 = check_num_images_table == real_count

newest_csv = new_csv[filter2]

newest_csv.to_csv(current_directory+"/../dataset.csv", index=False)