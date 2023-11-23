from itertools import groupby

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
        alpha=0.5,
        num_classes=86,
        features=None,
        center_file_path=None,
        name="ctc_center_loss",
        **kwargs):
        super(CTCCenterLoss, self).__init__(name=name, **kwargs)

        self.alpha = alpha
        self.num_classes = num_classes
        self.features = features
        self.centers = None

        if center_file_path is not None:
            # use tf2.0
            self.centers = tf.Variable(tf.io.read_file(center_file_path), trainable=False)
        else:
            self.centers = self.add_weight(
                name="centers",
                shape=[num_classes, features],
                initializer=tf.keras.initializers.RandomNormal(),
                dtype=tf.float32,
                trainable=False,
            )

    def call(self, y_true, y_pred):
        """获取center loss及center的更新op

        Arguments:
            features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
            labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
            alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
            num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.
            verbose: 打印中间过程

        Return:
            loss: Tensor,可与softmax loss相加作为总的loss进行优化.
            centers: Tensor,存储样本中心值的Tensor,仅查看样本中心存储的具体数值时有用.
            centers_update_op: op,用于更新样本中心的op,在训练时需要同时运行该op,否则样本中心不会更新
        """
        # 获取特征的维数，例如128维
        len_features = features.get_shape()[1]

        # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
        # 设置trainable=False是因为样本中心不是由梯度进行更新的
        centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)
        # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
        labels = tf.reshape(labels, [-1])
        print('tf.shape(labels):', tf.shape(labels))

        # 构建label
        # 根据样本label,获取mini-batch中每一个样本对应的中心值
        centers_batch = tf.gather(centers, labels)
        # 计算loss
        loss = tf.nn.l2_loss(features - centers_batch)

        # 当前mini-batch的特征值与它们对应的中心值之间的差
        diff = centers_batch - features

        # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff

        centers_update_op = tf.scatter_sub(centers, labels, diff)

        return loss, centers, centers_update_op


# main
if __name__ == '__main__':
    # check version of tensorflow
    print(tf.__version__)
    loss = CTCCenterLoss()
