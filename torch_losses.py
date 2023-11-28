import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CenterLoss(nn.Module):
    """
    Reference: Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    """

    def __init__(self, num_classes=86, feat_dim=96):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = torch.zeros([self.num_classes, self.feat_dim], dtype=torch.float32)

    def __call__(self, pred):
        features, predicts = pred

        feats_reshape = torch.reshape(features, [-1, features.shape[-1]])
        label = torch.argmax(predicts, axis=2)
        label = torch.reshape(label, [label.shape[0] * label.shape[1]]).float()

        batch_size = feats_reshape.shape[0]

        #calc l2 distance between feats and centers  
        _feat = torch.sum(torch.square(feats_reshape),
                                 axis=1,
                                 keepdim=True)
        _feat = torch.Tensor.expand(_feat, [batch_size, self.num_classes])

        _center = torch.sum(torch.square(self.centers), axis=1, keepdim=True)
        _center = torch.Tensor.expand(_center, [self.num_classes, batch_size]).float()
        _center = _center.T

        distmat = torch.add(_feat, _center)
        feat_dot_center = torch.matmul(feats_reshape, self.centers.T)
        distmat = distmat - 2.0 * feat_dot_center

        # generate the mask
        classes = torch.arange(self.num_classes).long()
        label = torch.Tensor.expand(torch.unsqueeze(label, 1), (batch_size, self.num_classes))
        mask = torch.Tensor.eq(
            torch.Tensor.expand(classes, [batch_size, self.num_classes]),
            label)
        dist = torch.multiply(distmat, mask)

        dist = torch.clip(dist, min=1e-12, max=1e+12)
        # mean
        loss = torch.mean(dist)
        return loss


if __name__ == '__main__':
    dims = 96
    loss = CenterLoss(num_classes=86, feat_dim=dims)
    # load from npy file
    x = np.load("features.npy")
    x = torch.from_numpy(x).float()
    labels = np.load("labels.npy")
    labels = torch.from_numpy(labels).long()

    print(x.max(), x.min())
    print(labels.max(), labels.min())
    l = loss([x, labels])
    print(l)