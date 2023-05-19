'''
LSTM Model for GuitarML

Source code from:
https://github.com/GuitarML/GuitarLSTM/blob/ee0983bb02af1e2db476466614d6421473beb01d/guitar_lstm_colab.ipynb
'''

import sys
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from scipy.io import wavfile

# Set training_mode and hyperparameters
train_mode = 0

if train_mode == 0:         # Speed Training
    learning_rate = 0.01 
    conv1d_strides = 12    
    conv1d_filters = 16
    hidden_units = 36
elif train_mode == 1:       # Accuracy Training (~10x longer than Speed Training)
    learning_rate = 0.01 
    conv1d_strides = 4
    conv1d_filters = 36
    hidden_units= 64
else:                       # Extended Training (~60x longer than Accuracy Training)
    learning_rate = 0.0005 
    conv1d_strides = 3
    conv1d_filters = 36
    hidden_units= 96


# WindowArray Dataset
class WindowArrayDataset(Dataset):
    def __init__(self, x, y, window_len):
        self.x = x
        self.y = y[window_len-1:]
        self.window_len = window_len

    def __len__(self):
        return len(self.x) - self.window_len + 1

    def __getitem__(self, index):
        x_out = self.x[index: index+self.window_len]
        y_out = self.y[index]
        return x_out, y_out


# Model Definition
class GuitarAmpEmulator(nn.Module):
    def __init__(self, conv1d_filters, conv1d_strides, hidden_units):
        super(GuitarAmpEmulator, self).__init__()
        self.conv1 = nn.Conv1d(1, conv1d_filters, 12, stride=conv1d_strides, padding=6)
        self.conv2 = nn.Conv1d(conv1d_filters, conv1d_filters, 12, stride=conv1d_strides, padding=6)
        self.lstm = nn.LSTM(conv1d_filters, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x.squeeze(1)


# Error to signal function
def error_to_signal(y_true, y_pred):
    y_true, y_pred = pre_emphasis_filter(y_true), pre_emphasis_filter(y_pred)
    return torch.sum(torch.pow(y_true - y_pred, 2), dim=0) / (torch.sum(torch.pow(y_true, 2), dim=0) + 1e-10)

# Pre-emphasis filter function
def pre_emphasis_filter(x, coeff=0.95):
    return torch.cat([x, x - coeff * x], 1)

# Save .wav file function
def save_wav(name, data):
    wavfile.write(name, 44100, data.flatten().astype(np.float32))

# Normalize function
def normalize(data):
    data_max = max(data)
    data_min = min(data)
    data_norm = max(data_max, abs(data_min))
    return data / data_norm

# Instantiate model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GuitarAmpEmulator(conv1d_filters, conv1d_strides, hidden_units).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Load and preprocess data
in_rate, in_data = wavfile.read(in_file)
out_rate, out_data = wavfile.read(out_file)

X_all = in_data.astype(np.float32).flatten()
X_all = normalize(X_all).reshape(len(X_all),1)
y_all = out_data.astype(np.float32).flatten()
y_all = normalize(y_all).reshape(len(y_all),1)

dataset = WindowArrayDataset(X_all, y_all, input_size)
train_examples = int(len(X_all) * 0.8)
train_set, val_set = random_split(dataset, [train_examples, len(dataset) - train_examples])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Train Model
epochs = 50
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x_batch.size(0)

    train_loss /= len(train_set)
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}')

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item() * x_batch.size(0)

    val_loss /= len(val_set)
    print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss}')

# Save Model
torch.save(model, f'/models/{name}/{name}.pth')

# show model structure
print(model)