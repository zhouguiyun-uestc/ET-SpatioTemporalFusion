from model import ETModel
from dataset import ETDataset
import torch
from tqdm import tqdm
from visdom import Visdom
from torch.utils.data import DataLoader
import matplotlib.pylab as plt
import numpy as np
import pickle


EPS = 1E-8

def compress(groundtruth, K=10, C=0.1):
    """
        Compress from (-inf, +inf) to [-K ~ K]
    """
    # if torch.is_tensor(groundtruth):
    #     #groundtruth = -100 * (groundtruth <= -100) + groundtruth * (groundtruth > -100)
    #     groundtruth = K * (1 - torch.exp(-C * groundtruth)) / (1 + torch.exp(-C * groundtruth))
    # else:
    #     #groundtruth = -100 * (groundtruth <= -100) + groundtruth * (groundtruth > -100)
    #     groundtruth = K * (1 - np.exp(-C * groundtruth)) / (1 + np.exp(-C * groundtruth))
    return groundtruth


def decompress(groundtruth, K=10, limit=9.9):
    # groundtruth = limit * (groundtruth >= limit) - limit * (groundtruth <= -limit) + groundtruth * (torch.abs(groundtruth) < limit)
    # groundtruth = -K * torch.log((K - groundtruth) / (K + groundtruth))
    return groundtruth


def save_pickle(name, data):
    with open(name, 'wb') as handle:
        pickle.dump(data, handle)


#有两个mask，一个用于处理无效数据，一个用于在训练时随机地某些数据置为无效
def mask_gen(groudtruth, sample_mask):
    mask1 = torch.zeros_like(groudtruth)
    mask1[torch.where(groudtruth != -1000.0)] = 1.0
    if sample_mask is not None:
        mask2 = torch.zeros_like(sample_mask)
        mask2[torch.where(sample_mask != 0.0)] = 1.0
    else:
        mask2 = None
    return mask1, mask2


def loss_func(pred, groundtruth, sample_mask, weight1, weight2, pretrain):
    pred_y = pred[:, -1]
    loss = torch.sum(weight1 * torch.abs(pred_y - groundtruth)) / (torch.sum(weight1) + EPS)
    pred_x = pred[:, :-1][..., :sample_mask.shape[-1]]
    loss += torch.sum(weight2 * torch.abs(pred_x - sample_mask)) / (torch.sum(weight2) + EPS)
    rmsd = torch.sum(weight1 * (decompress(pred_y) - groundtruth) ** 2) / (torch.sum(weight1) + EPS)
    return loss, rmsd


if __name__ == '__main__':
    root = "./Data"
    # Model config
    pretrain = False
    feature_dim = 7
    time_steps = 12
    hidden_dim = 128
    num_spatial = 3
    num_sequential = 2
    look_ahead = 1

    # Training config
    epoch = 5000
    num_per_epoch = 128  # 每一轮训练的次数
    batch_size = 2
    valid_freq = 5  # 每valid_freq次数后进行一次验证，计算验证集上的精度
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if pretrain:
        types = "ET_AVG"
        lr = 1e-4
    else:
        types = "groundtruth"
        lr = 1e-4

    data_loader_train = DataLoader(
        ETDataset("train", root=root, time_len=time_steps, look_ahead=look_ahead, lens=num_per_epoch, pretrain=pretrain,
                  type=types), batch_size=batch_size, num_workers=0, drop_last=True, pin_memory=True)
    data_loader_valid = DataLoader(
        ETDataset("valid", root=root, time_len=time_steps, look_ahead=look_ahead, lens=num_per_epoch, pretrain=pretrain,
                  type=types), batch_size=batch_size, num_workers=0, drop_last=True, pin_memory=True)
    model = ETModel(feature_dim=feature_dim,
                    time_steps=time_steps,
                    hidden_dim=hidden_dim,
                    num_spatial=num_spatial,
                    num_sequential=num_sequential,
                    look_ahead=look_ahead).to(device)

    save_path = 'checkpoint/stage1'
    if not pretrain:
        model.load_state_dict(torch.load(f"./checkpoint/stage2/model_epoch_avg_refined.pth")["model_param"])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    eval_metrics = {"train_loss": [], "valid_loss": [], "train_loss_sum": 0.0, "valid_loss_sum": 0.0}

    print("Start train!")

    for e in range(epoch):
        model.train()
        for sample, groudtruth, sample_mask in tqdm(data_loader_train):
            optimizer.zero_grad()
            try:
                pred = model(sample.float().to(device))
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print('WARNING: out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    else:
                        pass

            groudtruth = groudtruth.to(device)
            sample_mask = sample_mask.to(device)
            mask1, mask2 = mask_gen(groudtruth, sample_mask)
            loss, rmsd = loss_func(pred, groudtruth, sample_mask, mask1, mask2, pretrain)

            if pretrain:
                eval_metrics["train_loss"].append(loss.item())
                eval_metrics["train_loss_sum"] += loss.item()
            else:
                eval_metrics["train_loss"].append(rmsd.item())
                eval_metrics["train_loss_sum"] += rmsd.item()

            loss.backward()
            optimizer.step()

        if (e + 1) % valid_freq == 0:
            model.eval()
            with torch.no_grad():
                for sample, groudtruth, sample_mask in tqdm(data_loader_valid):
                    pred = model(sample.float().to(device))
                    groudtruth = groudtruth.to(device)

                    sample_mask = sample_mask.to(device)
                    mask1, mask2 = mask_gen(groudtruth, sample_mask)
                    loss, rmsd = loss_func(pred, groudtruth, sample_mask, mask1, mask2, pretrain)
                    if pretrain:
                        eval_metrics["valid_loss"].append(loss.detach().cpu().numpy())
                        eval_metrics["valid_loss_sum"] += loss.detach().cpu().numpy()
                    else:
                        eval_metrics["valid_loss"].append(rmsd.detach().cpu().numpy())
                        eval_metrics["valid_loss_sum"] += rmsd.detach().cpu().numpy()

                cur_train_loss = sum(eval_metrics['train_loss'][-valid_freq:]) / valid_freq
                cur_valid_loss = sum(eval_metrics['valid_loss'][-valid_freq:]) / valid_freq if len(
                    eval_metrics['valid_loss']) > 0 else 'nan'

                print(f"Epoch {e}: average train loss: {cur_train_loss}, average valid loss: {cur_valid_loss}")

        if (e + 1) % (valid_freq * 20) == 0:
            torch.save({'model_param': model.state_dict(), 'optim_param': optimizer.state_dict()},
                       f"./{save_path}/model_epoch_{e}_avg.pth")



        
