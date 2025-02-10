import torch

# Example usage:
if __name__ == '__main__':
    # Path to your CSV file with yfinance data
    csv_file = 'path_to_your_yfinance_data.csv'
    
    # Create an instance of the dataset (adjust the sequence_length and feature_columns as needed)
    dataset = YFinanceDataset(csv_file, sequence_length=60)
    
    # Create a DataLoader to handle batching and shuffling
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Example: Iterate over the DataLoader and print the shape of the sequences and targets
    for batch_idx, (sequences, targets) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Input sequences shape: {sequences.shape}")  # Expected: [batch_size, sequence_length, num_features]
        print(f"  Targets shape: {targets.shape}")            # Expected: [batch_size, num_features]
        # Optionally, break after one batch to verify
        break

