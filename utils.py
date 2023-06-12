import torch
import torch.nn as nn
import numpy as np

def batch_shuffle(batch_data):
    return batch_data[torch.randperm(batch_data.shape[0])]

class MeshLoss():
    def __init__(self): #layer_info_lst= [(point_num, feature_dim)]
        pass
    
    def compute_geometric_mean_euclidean_dist_error(self, gt_pc, predict_pc, weights = None):
        nbt, nfr, npt, nch = gt_pc.shape
        gt_pc = gt_pc.reshape(nbt * nfr, npt, nch)
        predict_pc = predict_pc.reshape(nbt * nfr, npt, nch)
        if weights is None:
            error = (gt_pc - predict_pc).pow(2).sum(dim=-1).pow(0.5).mean()
        else:
            error = ((gt_pc - predict_pc).pow(2).sum(dim=-1).pow(0.5) * weights.view(1, -1).to(gt_pc.device)).mean() * npt / torch.sum(weights)
        return error     

    def compute_geometric_maximal_euclidean_dist(self, gt_pc, predict_pc, weights = None):
        # B x T x P x 3
        # return ((gt_pc - predict_pc).pow(2).sum(dim=-1).pow(0.5)).max(dim=-1)[0].mean()
        nbt, nfr, npt, nch = gt_pc.shape
        gt_pc = gt_pc.reshape(nbt * nfr, npt, nch)
        predict_pc = predict_pc.reshape(nbt * nfr, npt, nch)
        if weights is None:
            error = (gt_pc - predict_pc).pow(2).sum(dim=-1).pow(0.5).max(dim=-1)[0].mean()
        else:
            error = ((gt_pc - predict_pc).pow(2).sum(dim=-1).pow(0.5) * weights.view(1, -1).to(gt_pc.device)).max(dim=-1)[0].mean()
        return error

    def compute_mesh_distance(self, mesh_a, mesh_b, mask=None):
        # mesh_a: (N, 3)
        # mesh_b: (N, 3)
        if mask is None:
            return np.sqrt(((mesh_a - mesh_b) ** 2).sum(axis=-1)).mean(axis=-1)
        else:
            return (np.sqrt(((mesh_a - mesh_b) ** 2).sum(axis=-1)) * mask).sum() / mask.sum()

    def dtw_error(self, seq_a, seq_b, mask = None):
        # seq_a: (T, N, 3)
        # seq_b: (T, N, 3)
        # seq_a = seq_a.detach().cpu().numpy()
        # seq_b = seq_b.detach().cpu().numpy()
        seq_a = seq_a.unsqueeze(1)
        seq_b = seq_b.unsqueeze(0)
        if mask is None:
            diff = (((seq_a - seq_b)**2).sum(dim = -1).pow(0.5)).mean(dim = -1)
        else:
            mask = mask.view(1, 1, -1)
            diff = (((seq_a - seq_b)**2).sum(dim = -1).pow(0.5) * mask).mean(dim = -1) * mask.shape[2] / torch.sum(mask)

        dist_lst = diff.cpu().detach().numpy()
        # print(dist_lst)
        # dist_lst = []
        # for i in range(seq_a.squeeze().shape[0]):
        #     for j in range(seq_b.squeeze().shape[0]):
        #         print(self.compute_mesh_distance(seq_a.squeeze()[i].cpu().numpy(), seq_b.squeeze()[j].cpu().numpy(), mask.squeeze().cpu().numpy()))
        #         print(dist_lst[i, j])

        # dist_lst = np.array(dist_lst)
        # dist_lst = dist_lst.reshape(seq_a.shape[0], seq_b.shape[0])

        #argmin = np.zeros_like(dist_lst)
        acc_dist = np.zeros_like(dist_lst)

        acc_dist[0, 0] = dist_lst[0, 0]
        for i in range(1, seq_b.shape[0]):
            acc_dist[0, i] = acc_dist[0, i - 1] + dist_lst[0, i]

        for i in range(1, dist_lst.shape[0]):
            acc_dist[i, 0] = acc_dist[i - 1, 0] + dist_lst[i, 0]
            for j in range(1, dist_lst.shape[1]):
                acc_dist[i, j] = min(acc_dist[i - 1, j], acc_dist[i, j - 1], acc_dist[i - 1, j - 1]) + dist_lst[i, j]
                #argmin[i, j] = np.argmin([acc_dist[i - 1, j], acc_dist[i, j - 1], acc_dist[i - 1, j - 1]])

        return acc_dist[-1, -1]