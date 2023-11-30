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

from tst import Transformer

from src.utils import compute_loss, fit, Logger
from src.metrics import MSE


# Training parameters
BATCH_SIZE = 10
NUM_WORKERS = 0
LR = 1e-3
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

reflectFile = 'traindata004/reflectivity4train.csv'

waveFile = 'traindata004/wavelet4train.csv'

seisFile = 'traindata004/seismic4train.csv'

seisDataset = SeisDataset(seisFile, reflectFile, waveFile) #

# Split between train, validation and test
dataset_train, dataset_val, dataset_test = random_split(
    seisDataset, (60000, 10000, 10601), generator=torch.Generator().manual_seed(0))

dataloader_train = DataLoader(dataset_train,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=False
                              )


dataloader_val = DataLoader(dataset_val,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=NUM_WORKERS
                            )

dataloader_test = DataLoader(dataset_test,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=NUM_WORKERS
                             )

net = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size,
                  dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)

# states = torch.load('models0528/transformer_15_2022_05_28__000035.pth')
# net.load_state_dict(states['net'])

summary(net,(1024,1))

optimizer = optim.Adam(net.parameters(), lr=LR)
loss_function = nn.MSELoss()

metrics = {
    'train_loss': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[-1], reduction='none'),
    'val_loss': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[-1], reduction='none'),
    'train_loss_r': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[-1], reduction='none'),
    'val_loss_r': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[-1], reduction='none'),
    'train_loss_w': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[-1], reduction='none'),
    'val_loss_w': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[-1], reduction='none')
}

logger = Logger('models/training.csv', model_name=net.name,
                params=[y for key in metrics.keys() for y in (key, key+'_std')])

# Fit model
with tqdm(total=EPOCHS) as pbar:
    loss = fit(net, optimizer, loss_function, dataloader_train,
               dataloader_val, epochs=EPOCHS, pbar=pbar, device=device)
# Save model
torch.save(net.state_dict(),
           'models/'+str(net.name)+'_'+str(datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S"))+'.pth')

# Switch to evaluation
# states = torch.load('models/transformer_9_2022_05_12__013503.pth')
# net.load_state_dict(states['net'])
_ = net.eval()

# Select target values in test split
r_true = seisDataset.ref_labels[dataloader_test.dataset.indices]
w_true = seisDataset.wav_labels[dataloader_test.dataset.indices]

# Compute predictions
predictions_r = torch.empty(len(dataloader_test.dataset), 1024, 1)
predictions_w = torch.empty(len(dataloader_test.dataset), 101, 1)
predictions1_r = torch.empty(len(dataloader_test.dataset), 1025)
predictions1_w = torch.empty(len(dataloader_test.dataset), 102)
rtrue = torch.empty(len(dataloader_test.dataset), 1025)
wtrue = torch.empty(len(dataloader_test.dataset), 102)
idx_prediction = 0

with torch.no_grad():
    for s, r, w in tqdm(dataloader_test, total=len(dataloader_test)):
        netout_r, netout_w = net(s.to(device))
        netout_r = netout_r.cpu()
        netout_w = netout_w.cpu()
        Preindices = torch.tensor(dataset_test.indices[idx_prediction:idx_prediction+s.shape[0]])
        Preindices = torch.unsqueeze(Preindices,dim=1)
        predictions_r[idx_prediction:idx_prediction + s.shape[0], ...] = netout_r
        predictions1_r[idx_prediction:idx_prediction + s.shape[0],...] = torch.cat([Preindices,torch.squeeze(netout_r, dim=2)], dim=1)
        predictions_w[idx_prediction:idx_prediction + s.shape[0], ...] = netout_w
        predictions1_w[idx_prediction:idx_prediction + s.shape[0], ...] = torch.cat([Preindices, torch.squeeze(netout_w, dim=2)], dim=1)
        idx_prediction += s.shape[0]

Rindices = torch.tensor(dataset_test.indices)
Rindices = torch.unsqueeze(Rindices,dim=1)
rtrue = torch.cat([Rindices,torch.squeeze(r_true, dim=2)], dim=1)
wtrue = torch.cat([Rindices,torch.squeeze(w_true, dim=2)], dim=1)
np.savetxt('models/test_R_True.csv', rtrue.numpy(),fmt='%.6f',delimiter=',')
np.savetxt('models/test_R_Pred.csv', predictions1_r.numpy(),fmt='%.6f',delimiter=',')
np.savetxt('models/test_W_True.csv', wtrue.numpy(),fmt='%.6f',delimiter=',')
np.savetxt('models/test_W_Pred.csv', predictions1_w.numpy(),fmt='%.6f',delimiter=',')

results_metrics = {
    key: value for key, func in metrics.items() for key, value in {
        key: func(r_true, predictions_r).mean(),
        key+'_std': func(w_true, predictions_w).std()
    }.items()
}

# Log
logger.log(**results_metrics)

fig = plt.figure()
ax = fig.add_subplot(311)
plt.plot(predictions_r[3, :, 0] , 'k')
plt.tight_layout()
ax = fig.add_subplot(312)
plt.plot(rtrue[3, 1:], 'r')
ax = fig.add_subplot(313)
plt.plot(predictions_r[3, :, 0] , 'k')
plt.tight_layout()
plt.plot(rtrue[3, 1:], 'r')
fig.savefig('models/rrr.png')

fig = plt.figure()
ax = fig.add_subplot(311)
plt.plot(predictions_w[3, :, 0] , 'k')
ax = fig.add_subplot(312)
plt.plot(wtrue[3, 1:], 'r')
ax = fig.add_subplot(313)
plt.plot(predictions_w[3, :, 0] , 'k')
plt.tight_layout()
plt.plot(wtrue[3, 1:], 'r')
fig.savefig('models/www.png')

print('prediction done!/n')
Loss_train_np = np.load('models/Loss_train.npy' )
Loss_val_np = np.load('models/Loss_val.npy')
Loss_train_r_np = np.load('models/Loss_train_r.npy' )
Loss_val_r_np = np.load('models/Loss_val_r.npy')
Loss_train_w_np = np.load('models/Loss_train_w.npy', )
Loss_val_w_np = np.load('models/Loss_val_w.npy')
Loss_train_s_np = np.load('models/Loss_train_s.npy')
Loss_val_s_np = np.load('models/Loss_val_s.npy')

fig= plt.figure()
x = range(0, EPOCHS)
ax = fig.add_subplot(221)
plt.plot(x, Loss_train_np, '.-', label='train loss')
plt.tight_layout()
plt.plot(x, Loss_val_np, '-', label='validation loss')
plt.xlabel('per epoch')
plt.ylabel('Total Loss')
plt.legend()

ax = fig.add_subplot(222)
plt.plot(x, Loss_train_r_np, '.-', label='train loss')
plt.tight_layout()
plt.plot(x, Loss_val_r_np, '-', label='validation loss')
plt.xlabel('per epoch')
plt.ylabel('Reflectivity Loss')
plt.legend()

ax = fig.add_subplot(223)
plt.plot(x, Loss_train_w_np, '.-', label='train loss')
plt.tight_layout()
plt.plot(x, Loss_val_w_np, '-', label='validation loss')
plt.xlabel('per epoch')
plt.ylabel('Wavelet Loss')
plt.legend()

ax = fig.add_subplot(224)
plt.plot(x, Loss_train_s_np, '.-', label='train loss')
plt.tight_layout()
plt.plot(x, Loss_val_s_np, '-', label='validation loss')
plt.xlabel('per epoch')
plt.ylabel('Synthetic Seismic Loss')
plt.legend()
plt.savefig('models/Loss.png')

print('done!/n')
