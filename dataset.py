from torch.utils.data import Dataset
import torch
from pathlib import Path
import numpy as np
import pandas as pd


class ETDataset(Dataset):
    def __init__(self, dataset, root, time_len, look_ahead, lens, pretrain = False, type = 'ET_AVG',
                 disturb=False, disturb_factor=0.1, disturb_feature_index=0):
        '''
        Dataset class for training the model
        '''
        self.dataset = dataset
        self.root = Path(root)
        self.time_len = time_len
        self.look_ahead = look_ahead
        self.lens = lens
        self.pretrain = pretrain
        self.type = type
        self.x, self.y = self.load_dataset(disturb, disturb_factor, disturb_feature_index)
        self.position = self.load_position()
        self.total_times = 0.0
        self.n = 0.0

    def load_position(self):
        df = pd.read_csv(self.root / "position.csv")
        df = df.drop_duplicates()
        df = df.values.tolist()
        return df

    def load_dataset(self, disturb=False, disturb_factor=0.1, disturb_feature_index=0):
        x = np.load(self.root / "feature.npy")

        if disturb is True:
            std_dev = np.std(x, axis=(1, 2))
            feature_std_dev = std_dev[disturb_feature_index, :, np.newaxis, np.newaxis]
            perturbation = feature_std_dev * disturb_factor
            perturbation = np.transpose(perturbation, (2, 1, 0))
            x[disturb_feature_index, :, :, :] += perturbation

        if self.pretrain:
            y = np.load(self.root / f"{self.type}.npy")
        else:
            y = np.load(self.root / f"groundtruth.npy")

        T = 288

        if self.pretrain and self.dataset == 'train':
            T = y.shape[-1]
            train_len = T
        elif self.dataset != 'inversion':
            if self.type == 'ET_avg':
                y = y[..., -12-T:-12]
            x = x[..., -12-T:-12]  # a[:,:,None] 和a[…, None]等价
        train_len = int(T * 0.8)

        if self.dataset == 'inversion':
            return x, y
        elif self.dataset == 'train':
            return x[..., :train_len], y[..., :train_len]
        elif self.dataset == 'valid':
            return x[..., train_len:], y[..., train_len:]
        elif self.dataset == 'test':
            return x[..., train_len:], y[..., train_len:]

    def sample(self, index, column_size, row_size, thresh):
        fs, cs, rs, T = self.x.shape
        times = 0.0
        x_gt = None
        drop_index = None
        mask = None
        if self.dataset == 'train' or self.dataset == 'valid':
            idx = torch.randint(low=0, high=len(self.position), size=(1,))
            pos = self.position[idx]
            col_range = int(0.1 * column_size)
            row_range = int(0.1 * row_size)

            col_len = torch.randint(low=-col_range, high=col_range, size=(1,))
            row_len = torch.randint(low=-row_range, high=row_range, size=(1,))

            col_left = int(0.5 * column_size) + col_len
            col_start = max(pos[0] - col_left, 0)
            if col_start + column_size > cs:
                col_end = cs
            else:
                col_end = col_start + column_size
            col_start = col_end - column_size

            row_left = int(0.5 * row_size) + row_len
            row_start = max(pos[1] - row_left, 0)
            if row_start + row_size > rs:
                row_end = rs
            else:
                row_end = row_start + row_size
            row_start = row_end - row_size

            x = self.x[:, col_start:col_end, row_start:row_end, :]
            y = self.y[col_start:col_end, row_start:row_end, :]

            t = torch.randint(low=0, high=T-self.look_ahead-self.time_len, size=(1,))
            x = x[..., t:t + self.time_len]
            y = y[..., t:t + self.time_len + self.look_ahead]

            valid_index = np.where(x != 0.0)
            drop_ratio = torch.rand((1,)) * 0.5
            drop_num = int(drop_ratio * len(valid_index[0]))
            if self.dataset == 'valid':
                drop_num = 0
            rand_index = torch.randint(len(valid_index), (drop_num,))
            drop_index = [vi[rand_index] for vi in valid_index]
            mask =np.zeros_like(x)
            try:
                mask[drop_index] = x[drop_index]
                x[drop_index] = 0.0
            except:
                pass
        
        else:
            t = index * self.time_len
            x = self.x[..., t:t + self.time_len]
            if self.dataset == 'inversion':
                y = self.y
            else:
                y = self.y[..., t:t + self.time_len + self.look_ahead]

        if x.shape[-1] < self.time_len:
            pad = np.zeros((fs, cs, rs, self.time_len-x.shape[-1]))
            x = np.concatenate([x, pad], axis=-1)

        if self.dataset != 'inversion' and y.shape[-1] < self.time_len + self.look_ahead:
            pad = -1000.0 * np.ones((cs, rs, self.time_len + self.look_ahead - y.shape[-1]))
            y = np.concatenate([y, pad], axis=-1)

        ocean_x, ocean_y, ocean_t = np.where(y == -1000.0)
        x[:, ocean_x, ocean_y, :] = -1000.0
        return x, y, mask, times

    def __getitem__(self, index):
        x, y, mask, times = self.sample(index, column_size=90, row_size=180, thresh=0.7)
        self.total_times += times
        self.n += 1.0
        if mask is None:
            return x, y
        else:
            return x, y, mask

    def __len__(self):
        if self.dataset == 'test' or self.dataset == 'inversion':
            return (self.x.shape[-1] + self.time_len - 1) // self.time_len
        else:
            return self.lens


