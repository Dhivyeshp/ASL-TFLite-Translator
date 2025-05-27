import os
import csv
import numpy as np
import pandas as pd

data_dir = "data"
output_file = "asl_dataset.csv"

rows = []

for label in os.listdir(data_dir):
    label_path = os.path.join(data_dir, label)
    if os.path.isdir(label_path):
        for file in os.listdir(label_path):
            if file.endswith(".csv"):
                with open(os.path.join(label_path, file), "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        rows.append(row + [label])  # add label as the last column

df = pd.DataFrame(rows)
df.to_csv(output_file, index=False, header=False)
print(f"Dataset saved to {output_file}")
