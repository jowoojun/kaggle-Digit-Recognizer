import os
import pandas as pd

class MnistDataset:
    def __init__(self, dataset_path, is_training=True):
        self.is_training = is_training

        # Load the data
        csv = pd.read_csv(dataset_path)

        if is_training == True:
            # Drop 'label' column
            self.data = csv.drop(labels = ["label"],axis = 1)
            self.label = csv["label"]
        else:
            self.data = csv
        del csv

        # Normalize the data
        self.data = self.data / 255.0



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_training == True:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

