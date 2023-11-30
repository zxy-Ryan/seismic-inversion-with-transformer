import csv
from pathlib import Path
import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
import matplotlib.pyplot as plt
from src.utils.utils import compute_loss
from tst.Sloss import SLoss


def fit(net, optimizer, loss_function, dataloader_train, dataloader_val, epochs=10, pbar=None, device='cpu'):
    val_loss_best = np.inf
    loss_function_R = torch.nn.L1Loss()
    loss_function_W = torch.nn.MSELoss()
    loss_function_S = SLoss()
    Loss_train = []
    Loss_val = []
    Loss_train_r = []
    Loss_val_r = []
    Loss_train_w = []
    Loss_val_w = []
    Loss_train_s = []
    Loss_val_s = []
    # Prepare loss history
    for idx_epoch in range(epochs):
        for idx_batch, (s, r, w) in enumerate(dataloader_train):
            optimizer.zero_grad()

            # Propagate input
            netout_r, netout_w = net(s.to(device))

            # Comupte loss
            rloss = loss_function_R(r.to(device), netout_r)

            wloss = loss_function_W(w.to(device), netout_w)

            sloss = loss_function_S(s.to(device), netout_r, netout_w)

            # loss = rloss/0.0012 + wloss/0.0286 +sloss/0.0037 # 0.004

            # loss = rloss/0.0046 + wloss/0.0505 +sloss/0.0246 # 0.002

            # loss = rloss/0.0046 + wloss/0.0271 # 0.004

            loss = rloss / 0.0046 + wloss / 0.0271 + idx_epoch * sloss / 0.013 / 190  # 0.004

            # Backpropage loss
            loss.backward()

            # Update weights
            optimizer.step()

        val_loss_r, val_loss_w, val_loss_s, val_loss = compute_loss(net, dataloader_val, loss_function, loss_function,
                                                                    loss_function_S, device)
        train_loss_r, train_loss_w, train_loss_s, train_loss = compute_loss(net, dataloader_train, loss_function,
                                                                            loss_function,
                                                                            loss_function_S, device)
        Loss_train.append(train_loss.item())
        Loss_val.append(val_loss.item())
        Loss_train_r.append(train_loss_r.item())
        Loss_val_r.append(val_loss_r.item())
        Loss_train_w.append(train_loss_w.item())
        Loss_val_w.append(val_loss_w.item())
        Loss_train_s.append(train_loss_s.item())
        Loss_val_s.append(val_loss_s.item())
        print(str(idx_epoch) + ' interation : total loss = '
              + str(train_loss.item()) + ';total val_loss = ' + str(val_loss.item())
              + '; train_loss_r = ' + str(train_loss_r.item()) + ';  val_loss_r = ' + str(val_loss_r.item())
              + '; train_loss_w = ' + str(train_loss_w.item()) + ';  val_loss_w = ' + str(val_loss_w.item())
              + '; train_loss_s = ' + str(train_loss_s.item()) + ';  val_loss_s = ' + str(val_loss_s.item()) + '/n')

        all_states = {"net": net.state_dict(), "optimizer": optimizer.state_dict(), "epoch": idx_epoch}
        torch.save(obj=all_states, f='models/' + str(net.name) + '_' + str(idx_epoch) + '_' + str(
            datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S")) + '.pth')

        if val_loss < val_loss_best:
            val_loss_best = val_loss

        if pbar is not None:
            pbar.update()

    Loss_train_np = np.array(Loss_train)
    Loss_val_np = np.array(Loss_val)
    Loss_train_r_np = np.array(Loss_train_r)
    Loss_val_r_np = np.array(Loss_val_r)
    Loss_train_w_np = np.array(Loss_train_w)
    Loss_val_w_np = np.array(Loss_val_w)
    Loss_train_s_np = np.array(Loss_train_s)
    Loss_val_s_np = np.array(Loss_val_s)

    np.save('models/Loss_train.npy', Loss_train_np)
    np.save('models/Loss_val.npy', Loss_val_np)
    np.save('models/Loss_train_r.npy', Loss_train_r_np)
    np.save('models/Loss_val_r.npy', Loss_val_r_np)
    np.save('models/Loss_train_w.npy', Loss_train_w_np)
    np.save('models/Loss_val_w.npy', Loss_val_w_np)
    np.save('models/Loss_train_s.npy', Loss_train_s_np)
    np.save('models/Loss_val_s.npy', Loss_val_s_np)

    return val_loss_best


def kfold(dataset, n_chunk, batch_size, num_workers):
    indexes = np.arange(len(dataset))
    chunks_idx = np.array_split(indexes, n_chunk)

    for idx_val, chunk_val in enumerate(chunks_idx):
        chunk_train = np.concatenate(
            [chunk_train for idx_train, chunk_train in enumerate(chunks_idx) if idx_train != idx_val])

        subset_train = Subset(dataset, chunk_train)
        subset_val = Subset(dataset, chunk_val)

        dataloader_train = DataLoader(subset_train,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers
                                      )
        dataloader_val = DataLoader(subset_val,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers
                                    )

        yield dataloader_train, dataloader_val


def leargnin_curve(dataset, n_part, validation_split, batch_size, num_workers):
    # Split train and val
    val_split = int(len(dataset) * validation_split)
    subset_train, subset_val = random_split(dataset, [len(dataset) - val_split, val_split])

    dataloader_val = DataLoader(subset_val,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers
                                )

    for idx in np.linspace(0, len(subset_train), n_part + 1).astype(int)[1:]:
        subset_learning = Subset(dataset, subset_train.indices[:idx])
        dataloader_train = DataLoader(subset_learning,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers
                                      )

        yield dataloader_train, dataloader_val


class Logger:
    def __init__(self, csv_path, model_name='undefined', params=[]):
        csv_path = Path(csv_path)

        if csv_path.is_file():
            self.csv_file = open(csv_path, 'a')
            self.writer = csv.DictWriter(self.csv_file, ['date', 'model'] + params)
        else:
            self.csv_file = open(csv_path, 'w')
            self.writer = csv.DictWriter(self.csv_file, ['date', 'model'] + params)
            self.writer.writeheader()

        self.model_name = model_name

    def log(self, **kwargs):
        kwargs.update({
            'date': datetime.datetime.now().isoformat(),
            'model': self.model_name
        })
        self.writer.writerow(kwargs)
        self.csv_file.flush()

    def __del__(self):
        self.csv_file.close()
