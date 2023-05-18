import torch
import scipy
import numpy as np

from model import TCN, causal_crop
from config import *

model.eval()
x_pad = torch.nn.functional.pad(x_batch, (rf-1, 0))
with torch.no_grad():
  y_hat = model(x_pad, c)

input = causal_crop(x_batch.view(-1).detach().cpu().numpy(), y_hat.shape[-1])
output = y_hat.view(-1).detach().cpu().numpy()
target = causal_crop(y_batch.view(-1).detach().cpu().numpy(), y_hat.shape[-1])

# apply highpass to output
sos = scipy.signal.butter(8, 20.0, fs=sample_rate, output="sos", btype="highpass")
output = scipy.signal.sosfilt(sos, output)

input /= np.max(np.abs(input))
output /= np.max(np.abs(output))
target /= np.max(np.abs(target))

# Calculate Mean Squared Error (MSE)
mse = np.mean((output - target) ** 2)

# Print or use the MSE value as needed
print(f"MSE: {mse}")