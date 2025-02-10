import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class YFinanceDataset(Dataset):
    def __init__(self, csv_file, sequence_length=60, feature_columns=None):
        """
        Args:
            csv_file (str): Path to the CSV file containing yfinance data.
            sequence_length (int): Number of time steps to include in each input sequence.
            feature_columns (list or None): Specific columns to use as features.
                                             If None, use all columns except 'Date' if present.
        """
        # Read the CSV file into a DataFrame
        self.data = pd.read_csv(csv_file)
        
        # If specific feature columns are provided, select them; otherwise, drop a 'Date' column if it exists
        if feature_columns is not None:
            self.data = self.data[feature_columns]
        else:
            if 'Date' in self.data.columns:
                self.data = self.data.drop(columns=['Date'])
        
        # Convert the DataFrame values to a NumPy array of type float32
        self.data_values = self.data.values.astype('float32')
        
        # Store the desired sequence length
        self.sequence_length = sequence_length
        
    def __len__(self):
        # The number of sequences is the total number of rows minus the sequence length.
        return len(self.data_values) - self.sequence_length

    def __getitem__(self, index):
        """
        For a given index, return a tuple (input_sequence, target) where:
        - input_sequence: A sequence of length 'sequence_length' of consecutive data points.
        - target: The data point immediately following the sequence.
        """
        # Extract a sequence of data for the input
        sequence = self.data_values[index : index + self.sequence_length]
        # Define the target as the data point right after the sequence
        target = self.data_values[index + self.sequence_length]
        
        # Convert both the sequence and target to PyTorch tensors
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        
        return sequence_tensor, target_tensor


