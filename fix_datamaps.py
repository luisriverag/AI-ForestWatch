import pickle
import os

# folder where your .pkl files live
split_dir = "Data/train/temp_split"

# the three files to fix
files = ["train_datamap.pkl", "val_datamap.pkl", "test_datamap.pkl"]

old_prefix = "/home/sukhan/Scratch/AI-Forest Dataset"
new_prefix = "Data/train"

for fname in files:
    path = os.path.join(split_dir, fname)
    # 1) load
    with open(path, "rb") as f:
        datamap = pickle.load(f)
    
    # datamap is a tuple: (list_of_paths, list_of_(path, label1, label2))
    list_paths, list_tuples = datamap

    # 2) replace in both the flat list and inside each tuple
    new_list_paths = [p.replace(old_prefix, new_prefix) for p in list_paths]
    new_list_tuples = [
        (p.replace(old_prefix, new_prefix), a, b)
        for (p, a, b) in list_tuples
    ]

    # 3) write back
    with open(path, "wb") as f:
        pickle.dump((new_list_paths, new_list_tuples), f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✔ Updated {fname}")
