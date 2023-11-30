import datetime
import torch
torch.cuda.empty_cache()
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary
import time
from tst import Transformer


# Training parameters
BATCH_SIZE = 10
NUM_WORKERS = 0
LR = 2e-3
EPOCHS = 10

# Model parameters
d_model = 32  # Lattent dim
q = 8  # Query size
v = 8  # Value size
h = 8  # Number of heads
N = 2  # Number of encoder and decoder to stack
attention_size = 12  # Attention window size
dropout = 0.2  # Dropout rate
pe = None  # Positional encoding
chunk_mode = None

d_input = 1  # From dataset
d_output = 1  # From dataset

netFile = 'output/transformer_19_2022_05_26__111123.pth' 


# NonStationary prediction
reflectFile = 'test4NS/reflectivity4testNS.csv'
waveFile = 'test4NS/wavelet4testNS.csv'
seisFile = 'test4NS/seismic4testNS.csv'

reflectoutFile = 'test4NSout/test_R_Pred.csv'
waveoutFile = 'test4NSout/test_W_Pred.csv'

rimageFile = 'test4NSout/rrr.png'
wimageFile = 'test4NSout/www.png'

# reflectFile = 'test4real/reflectivity4testS.csv'
# waveFile = 'test4real/wavelet4testS.csv'
# seisFile = 'test4real/seismic4testS.csv'
#
# reflectoutFile = 'test4realout/test_R_Pred3.csv'
# waveoutFile = 'test4realout/test_W_Pred3.csv'
#
# rimageFile = 'test4realout/rrr3.png'
# wimageFile = 'test4realout/www3.png'

# reflectFile = 'test4S/reflectivity4testS.csv'
# waveFile = 'test4S/wavelet4testS.csv'
# seisFile = 'test4S/seismic4testS.csv'
#
# reflectoutFile = 'test4TBI/test_R_Pred.csv'
# waveoutFile = 'test4TBI/test_W_Pred.csv'
#
# rimageFile = 'test4TBI/rrr.png'
# wimageFile = 'test4TBI/www.png'

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

class SeisDataset(Dataset):
    def __init__(self, seisFile, reflectFile, waveFile, transform=None, target_transform=None):

        self.ref_labels = np.genfromtxt(reflectFile, delimiter=',')
        self.ref_labels = torch.from_numpy(self.ref_labels)
        self.ref_labels = self.ref_labels.unsqueeze(2)
        self.ref_labels = self.ref_labels.to(torch.float32)

        self.wav_labels = np.genfromtxt(waveFile, delimiter=',')
        self.wav_labels = torch.from_numpy(self.wav_labels)
        self.wav_labels = self.wav_labels.unsqueeze(2)
        self.wav_labels = self.wav_labels.to(torch.float32)

        self.seis_samples = np.genfromtxt(seisFile, delimiter=',')
        self.seis_samples = torch.from_numpy(self.seis_samples)
        self.seis_samples = self.seis_samples.unsqueeze(2)
        self.seis_samples = self.seis_samples.to(torch.float32)

    def __len__(self):
        return np.shape(self.ref_labels)[0]

    def __getitem__(self, idx):

        reflabel = self.ref_labels[idx,...]
        wavelabel = self.wav_labels[idx,...]
        seisSample = self.seis_samples[idx,...]

        return seisSample, reflabel, wavelabel

seisDataset = SeisDataset(seisFile, reflectFile, waveFile) #

dataloader_test = DataLoader(seisDataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=NUM_WORKERS,
                              pin_memory=False
                              )

# Load transformer with Adam optimizer and MSE loss function
net = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size,
                  dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)

states = torch.load(netFile)
net.load_state_dict(states['net'])

summary(net,(1024,1))

# Switch to evaluation
_ = net.eval()

# Select target values in test split
r_true = seisDataset.ref_labels
w_true = seisDataset.wav_labels

# Compute predictions
predictions_r = torch.empty(len(dataloader_test.dataset), 1024)
predictions_w = torch.empty(len(dataloader_test.dataset), 101)

idx_prediction = 0
time_start = time.time()
with torch.no_grad():
    for s, r, w in tqdm(dataloader_test, total=len(dataloader_test)):
        netout_r, netout_w = net(s.to(device))
        netout_r = netout_r.cpu()
        netout_w = netout_w.cpu()
        predictions_r[idx_prediction:idx_prediction + s.shape[0], ...] = torch.squeeze(netout_r, dim=2)
        predictions_w[idx_prediction:idx_prediction + s.shape[0], ...] = torch.squeeze(netout_w, dim=2)
        idx_prediction += s.shape[0]
time_end = time.time()
print('Using time: ')
print(time_end-time_start)

np.savetxt(reflectoutFile, predictions_r.numpy(),fmt='%.6f',delimiter=',')
np.savetxt(waveoutFile, predictions_w.numpy(),fmt='%.6f',delimiter=',')

fig = plt.figure()
ax = fig.add_subplot(311)
plt.plot(predictions_r[3, :] , 'k')
plt.tight_layout()
ax = fig.add_subplot(312)
plt.plot(r_true[3, 1:], 'r')
ax = fig.add_subplot(313)
plt.plot(predictions_r[3, :] , 'k')
plt.tight_layout()
plt.plot(r_true[3, 1:], 'r')
fig.savefig(rimageFile)

fig = plt.figure()
ax = fig.add_subplot(311)
plt.plot(predictions_w[3, :] , 'k')
ax = fig.add_subplot(312)
plt.plot(w_true[3, 1:], 'r')
ax = fig.add_subplot(313)
plt.plot(predictions_w[3, :] , 'k')
plt.tight_layout()
plt.plot(w_true[3, 1:], 'r')
fig.savefig(wimageFile)

print('done!/n')
