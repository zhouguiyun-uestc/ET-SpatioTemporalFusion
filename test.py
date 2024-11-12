from model import ETModel
from dataset import ETDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle

torch.cuda.empty_cache()

EPS = 1E-8


def load_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def save_pickle(name, data):
    with open(name, 'wb') as handle:
        pickle.dump(data, handle)


def mask_gen(pred, groudtruth):
    mask = np.zeros_like(groudtruth)
    mask[np.where(groudtruth != -1000.0)] = 1.0
    return mask


def coeff(pred, groundtruth, mask):
    pred = pred[np.where(mask == 1.0)]
    groundtruth = groundtruth[np.where(mask == 1.0)]
    mean_pred = np.mean(pred)
    mean_gt = np.mean(groundtruth)
    print(f"mean_pred: {mean_pred}, mean_gt: {mean_gt}")
    cov = np.mean((pred - mean_pred) * (groundtruth - mean_gt))
    std_pred = np.sqrt(np.mean((pred - mean_pred)**2))
    std_gt = np.sqrt(np.mean((groundtruth - mean_gt)**2))
    coef = cov / (std_pred * std_gt)
    print(f"coefficent: {coef**2}, std_pred: {std_pred}, std_gt: {std_gt}")


def loss_func(pred, groudtruth, weight):
    loss = np.sum(weight * (pred - groudtruth) ** 2) / (np.sum(weight) + EPS)
    print(f"RMSD: {np.sqrt(loss)}")
    return loss


def test(pth_path, hidden_dim=512, disturb=False, disturb_factor=0.1, disturb_feature_index=0,):
    root = "./Data"
    # Model config
    feature_dim = 7
    time_steps = 12
    hidden_dim = hidden_dim
    num_spatial = 3
    num_sequential = 2
    look_ahead = 1

    # Training config
    num_per_epoch = 128
    batch_size = 1
    n_workers = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_dataset = ETDataset("test", root=root, time_len=time_steps, look_ahead=look_ahead, lens=num_per_epoch,
                             type="groundtruth", disturb=disturb, disturb_factor=disturb_factor,
                             disturb_feature_index=disturb_feature_index)

    data_loader_test = DataLoader(test_dataset,
                                    batch_size=batch_size, \
                                    num_workers=n_workers, drop_last=True, pin_memory=True)
    
    model = ETModel(feature_dim = feature_dim, 
                     time_steps = time_steps, 
                     hidden_dim = hidden_dim, 
                     num_spatial = num_spatial, 
                     num_sequential = num_sequential,
                     look_ahead = look_ahead).to(device)
    model.load_state_dict(torch.load(pth_path)["model_param"])
    true_lens = test_dataset.x.shape[-1]
    print(true_lens)

    preds = []
    gts = []
    print("Start test!")
    with torch.no_grad():
        model.eval()
        for sample, groundtruth in tqdm(data_loader_test):
            pred = model(sample.float().to(device)).cpu()[..., :-look_ahead][-1]
            groundtruth = groundtruth.squeeze(dim=0)[..., :-look_ahead]
            print(pred.shape, groundtruth.shape)

            pred = pred.detach().numpy()
            groundtruth = groundtruth.numpy()
            preds.append(pred)
            gts.append(groundtruth)
    
    preds = np.concatenate(preds, axis=-1)[..., :true_lens]
    gts = np.concatenate(gts, axis=-1)[..., :true_lens]

    mask = mask_gen(preds, gts)
    rmsd = loss_func(preds, gts, mask)
    coeff(preds, gts, mask)

    preds = preds[np.where(mask == 1.0)].flatten()
    gts = gts[np.where(mask == 1.0)].flatten()

    return preds, gts
