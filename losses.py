from itertools import groupby

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import Layer


class CenterLoss(Layer):
    def __init__(self, alpha=0.5, name="center_loss", **kwargs):
        super(CenterLoss, self).__init__(name=name, **kwargs)
        self.alpha = alpha
        self.centers = None
        
    def build(self, input_shape):
        self.centers = self.add_weight(
            name="centers",
            shape=[input_shape[1], input_shape[2]],
            initializer=tf.keras.initializers.RandomNormal(),
            dtype=tf.float32,
            trainable=False,
        )
        super(CenterLoss, self).build(input_shape)

    def call(self, y_true, y_pred):
        # compute center loss
        delta_centers = tf.gather(self.centers, indices=y_true, axis=0) - y_pred
        centers_counts = tf.math.bincount(y_true)
        centers_counts = tf.gather(centers_counts, indices=y_true, axis=0)
        delta_centers = delta_centers / (centers_counts[:, None] + 1)
        delta_centers = self.alpha * delta_centers
        self.centers.assign_sub(delta_centers)
        # compute loss
        loss = tf.reduce_mean(tf.square(y_pred - tf.gather(self.centers, indices=y_true, axis=0)))
        return loss


class FocalCTCLoss(Layer):
    def __init__(self, alpha=2.0, gamma=3.0, name="focal_ctc_loss", **kwargs):
        super(FocalCTCLoss, self).__init__(name=name, **kwargs)
        self.loss_fn = K.ctc_batch_cost
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        input_length = K.tile([[K.shape(y_pred)[1]]], [K.shape(y_pred)[0], 1])
        label_length = K.tile([[K.shape(y_true)[1]]], [K.shape(y_true)[0], 1])
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        p = tf.exp(-loss)
        focal_ctc_loss = tf.multiply(tf.multiply(self.alpha, tf.pow((1 - p), self.gamma)), loss)
        return tf.reduce_mean(focal_ctc_loss)


class CELoss(Layer):
    def __init__(self, name="ce_loss"):
        super(CELoss, self).__init__(name=name)

    def call(self, y_true, y_pred, **kwargs):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE,
        )(y_true, y_pred)
        return loss


class BCELoss(Layer):
    def __init__(self, name="bce_loss"):
        super(BCELoss, self).__init__(name=name)

    def call(self, y_true, y_pred, **kwargs):
        loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE,
        )(y_true, y_pred)
        return loss


class SmoothL1Loss:
    def __init__(self, name="smooth_l1_loss"):
        super(SmoothL1Loss, self).__init__(name=name)

    """ Compute smooth l1 loss between the predicted bounding boxes and the ground truth bounding boxes.

    Args:
        - y_true: The ground truth bounding boxes.
        - y_pred: The predicted bounding boxes.
    """

    def call(self, y_true, y_pred, **kwargs):
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)


class DiceLoss(Layer):
    def __init__(self, name="dice_loss"):
        super(DiceLoss, self).__init__(name=name)

    def call(self, y_true, y_pred, **kwargs):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
        dice = tf.reduce_mean((2.0 * intersection + 1e-7) / (union + 1e-7))
        return 1 - dice


class IOULoss(Layer):
    def __init__(self, name="iou_loss"):
        super(IOULoss, self).__init__(name=name)

    def call(self, y_true, y_pred, **kwargs):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
        iou = tf.reduce_mean((intersection + 1e-7) / (union - intersection + 1e-7))
        return 1 - iou


class DiceBCELoss(Layer):
    def __init__(self, name="dice_bce_loss"):
        super(DiceBCELoss, self).__init__(name=name)

    def call(self, y_true, y_pred, **kwargs):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # flatten label and prediction tensors
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice_loss = 1 - (2. * intersection + 1e-7) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-7)

        bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE,
        )(y_true, y_pred)

        return dice_loss + bce


class CTCLoss(Layer):
    def __init__(self, name="ctc_loss"):
        super(CTCLoss, self).__init__(name=name)

    def call(self, y_true, y_pred, **kwargs):
        input_length = K.tile([[K.shape(y_pred)[1]]], [K.shape(y_pred)[0], 1])
        label_length = K.tile([[K.shape(y_true)[1]]], [K.shape(y_true)[0], 1])
        loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss


# ctc center loss
class CTCCenterLoss(Layer):
    def __init__(self, 
        alpha=0.05,
        num_classes=86,
        feat_dims=128,
        random_init=True,
        name="ctc_center_loss",
        **kwargs):
        super(CTCCenterLoss, self).__init__(name=name, **kwargs)

        self.alpha = alpha
        self.num_classes = num_classes
        self.feat_dims = feat_dims
        self.centers = tf.Variable(
            initial_value=tf.zeros(shape=(num_classes, feat_dims), dtype=tf.float32, name="centers"),
            trainable=False,
            name="centers",
        )
        # random init centers
        if random_init:
            self.centers.assign(tf.random.normal(shape=(num_classes, feat_dims), mean=0.0, stddev=1.0))

    def call(self, y_true, y_pred, **kwargs):
        """
        Args:
            y_pred: [batch_size, seq_len, feat_dims]
            y_true: [batch_size]
        """
        _, _, feat_dims = tf.shape(y_pred)
        y_pred = tf.reshape(y_pred, shape=(-1, feat_dims))
        y_true = tf.reshape(y_true, shape=(-1, ))
        bs = tf.shape(y_true)[0]

        distmat = tf.pow(y_pred, 2)
        distmat = tf.reduce_sum(distmat, axis=1, keepdims=True)
        # expand to [bs, num_classes]
        distmat = tf.tile(distmat, multiples=(1, self.num_classes))
        distmat = tf.subtract(distmat, 2 * tf.matmul(y_pred, tf.transpose(self.centers)))
        distmat = tf.add(distmat, tf.transpose(tf.reduce_sum(tf.pow(self.centers, 2), axis=1, keepdims=True)))

        # mask
        classes = tf.range(self.num_classes, dtype=tf.int32)
        labels = tf.tile(tf.expand_dims(y_true, axis=1), multiples=(1, self.num_classes))
        mask = tf.math.equal(labels, classes)
        mask = tf.cast(mask, dtype=tf.float32)

        # compute loss
        dist = tf.multiply(distmat, mask)
        # clamp dist
        dist = tf.clip_by_value(dist, clip_value_min=1e-12, clip_value_max=1e+12)
        loss = tf.reduce_sum(dist) / tf.cast(bs, dtype=tf.float32)

        return loss


# main
if __name__ == '__main__':
    # check version of tensorflow
    print(tf.__version__)
    dims = 96
    loss = CTCCenterLoss(num_classes=86, feat_dims=dims)
    features = tf.random.normal(shape=(32, 16, dims))
    # save as npy
    np.save("features.npy", features.numpy())
    # labels: int32
    labels = tf.random.uniform(shape=(32, 16), minval=0, maxval=86, dtype=tf.int32)
    # save as npy
    np.save("labels.npy", labels.numpy())
    l = loss(labels, features)
    print(l)
