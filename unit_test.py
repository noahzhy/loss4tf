import os
import sys
import time
import random

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

from losses import CTCCenterLoss
from torch_losses import CenterLoss


# test ctccenterloss between torch and tf
def test_ctc_center_loss(pred, target, n_class=100, dims=96):
    # data
    features = tf.random.normal(shape=(32, 16, dims), dtype=tf.float32)
    labels = tf.random.uniform(shape=(32, 16, 86), minval=0, maxval=86, dtype=tf.float32)

    torch_loss = CenterLoss(num_classes=n_class, feat_dim=dims)([features, labels])
    torch_loss = torch_loss.detach().numpy().astype(np.float32)

    c_loss = CTCCenterLoss(num_classes=n_class, feat_dims=dims, random_init=False)
    tf_loss = c_loss(0, [features, labels])
    tf_loss = tf_loss.numpy().astype(np.float32)

    # compare
    print("torch loss:", torch_loss)
    print("tf    loss:", tf_loss)
    print("loss  diff:", torch_loss - tf_loss)


if __name__ == '__main__':
    n_class = 100
    dims = 128
    x = np.random.normal(size=(32, 16, dims)).astype(np.float32)
    labels = np.random.randint(0, n_class, (32, 16)).astype(np.int32)
    test_ctc_center_loss(x, labels, n_class, dims)
