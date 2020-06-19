import torch
import torch.nn as nn
from torch.autograd import Variable 

class LSTM(nn.Module):
  def __init__(self, input_size, output_size):
    super(LSTM, self).__init__()

    # Hidden dimensions
    self.hidden_dim = 64

    # Number of hidden layers
    self.layer_dim = 2

    # LSTM
    # batch_first=True (batch_dim, seq_dim, feature_dim)
    self.lstm = nn.LSTM(input_size, self.hidden_dim, self.layer_dim, batch_first = True, dropout = 0.5)

    # Readout layer
    self.fc = nn.Linear(self.hidden_dim, output_size)

  def forward(self, x):
    # x shape (batch, time_step, input_size)
    # r_out shape (batch, time_step, output_size)
    # h_n shape (n_layers, batch, hidden_size)
    # h_c shape (n_layers, batch, hidden_size)
    r_out, (h_n, h_c) = self.lstm(x, None)   # None represents zero initial hidden state

    # choose r_out at the last time step
    out = self.fc(r_out[:, -1, :])
    return out
