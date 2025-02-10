import torch
import torch.nn as nn

class MyLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Args:
            input_size (int): Number of features in the input data.
            hidden_size (int): Number of features in the hidden state.
            output_size (int): Dimension of the output.
            num_layers (int): Number of stacked LSTM layers.
        """
        super(MyLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM layer. Using batch_first=True allows inputs to be shaped as (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define a fully connected layer to map the hidden state to the final output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            torch.Tensor: Output tensor, for example of shape (batch_size, output_size)
        """
        # x is of shape (batch_size, sequence_length, input_size)
        batch_size = x.size(0)

        # Optionally, initialize hidden and cell states.
        # They are of shape (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        # Pass through the LSTM layer
        # out has shape (batch_size, sequence_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # For many tasks, we only need the output from the last time step.
        last_time_step_output = out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Map the LSTM output to the final output dimension
        output = self.fc(last_time_step_output)
        return output

