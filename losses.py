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
        feat_dims=96,
        random_init=False,
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

        features, preds = y_pred[0], y_pred[1]
        feats_reshape = tf.reshape(features, shape=(-1, self.feat_dims))
        label = tf.argmax(preds, axis=-1)
        label = tf.reshape(label, shape=(tf.shape(label)[0] * tf.shape(label)[1],))

        bs = tf.shape(feats_reshape)[0]

        feat = tf.reduce_sum(tf.pow(feats_reshape, 2), axis=1, keepdims=True)
        feat = tf.broadcast_to(feat, shape=(bs, self.num_classes))

        center = tf.reduce_sum(tf.pow(self.centers, 2), axis=1, keepdims=True)
        center = tf.broadcast_to(center, shape=(self.num_classes, bs))
        center = tf.cast(center, dtype=tf.float32)
        center = tf.transpose(center)

        distmat = tf.add(feat, center)

        feat_dot_center = tf.matmul(feats_reshape, tf.transpose(self.centers))
        distmat = distmat - 2.0 * feat_dot_center

        # mask
        classes = tf.range(self.num_classes, dtype=tf.int64)
        label = tf.broadcast_to(tf.expand_dims(label, axis=1), shape=(bs, self.num_classes))
        mask = tf.math.equal(
            tf.broadcast_to(classes, shape=(bs, self.num_classes)),
            label)
        mask = tf.cast(mask, dtype=tf.float32)

        # compute loss
        dist = tf.multiply(distmat, mask)

        # clamp dist
        dist = tf.clip_by_value(dist, clip_value_min=1e-12, clip_value_max=1e+12)
        loss = tf.reduce_mean(dist)

        return loss


# main
if __name__ == '__main__':
    # check version of tensorflow
    print(tf.__version__)
    dims = 96
    loss = CTCCenterLoss()
    features = tf.random.normal(shape=(32, 16, dims), dtype=tf.float32)
    np.save("features.npy", features.numpy())

    labels = tf.random.uniform(shape=(32, 16, 86), minval=0, maxval=86, dtype=tf.float32)
    np.save("labels.npy", labels.numpy())
    # f_c = tf.concat([features, labels], axis=-1)
    l = loss(0, [features, labels])
    print(l)
