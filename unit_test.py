import os
import sys
import time
import random

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

from losses import CTCCenterLoss
from torch_test import CenterLoss


# test ctccenterloss between torch and tf
def test_ctccenterloss(pred, target):
    dims = 96
    # torch
    x = torch.from_numpy(pred).reshape(32 * 16, dims)
    labels = torch.from_numpy(target).reshape(32 * 16, )
    torch_loss = CenterLoss(num_classes=86, feat_dim=dims, random_init=False, use_gpu=False)(x, labels)
    # to numpy float32
    torch_loss = torch_loss.detach().numpy().astype(np.float32)

    # tf
    tf_loss = CTCCenterLoss(num_classes=86, feat_dims=dims, random_init=False)(pred, target)
    # to numpy float32
    tf_loss = tf_loss.numpy().astype(np.float32)

    # compare
    print("torch loss:", torch_loss)
    print("tf loss:", tf_loss)
    print("loss diff:", torch_loss - tf_loss)


if __name__ == '__main__':
    # load from npy file
    x = np.load("features.npy")
    labels = np.load("labels.npy")
    test_ctccenterloss(x, labels)
