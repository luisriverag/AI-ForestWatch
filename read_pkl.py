import pickle
import os

# print(len(os.listdir('Data/train/training_2015_pickled_data')))

# Directory containing the pickle files
base_path = "Data/train/temp_split"

# List of files to read
files = ["train_datamap.pkl", "val_datamap.pkl", "test_datamap.pkl"]

# Read and print each file
for file in files:
    file_path = os.path.join(base_path, file)
    print(f"\nReading: {file_path}")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        print(f"{file} contents:\n", data)
