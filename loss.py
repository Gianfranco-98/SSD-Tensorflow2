#!/usr/bin/env python


# Stock libraries
import math
import warnings
import tensorflow as tf
from tensorflow.keras.losses import Huber
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# My libraries
from configuration import *
from detection_tools import *


class IoU_Loss:
    """
    Class handling the losses based on the IoU 
    """
    def __init__(self, loss_type):
        """
        IoU class constructor

        Parameters
        ----------
        loss_type: which type of iou loss to compute
        """
        self.loss_type = loss_type

    def RDIoU(self, gt_bboxes, pred_bboxes):
        """
        Compute the penalty term RDIoU

        Parameters 
        ----------
        gt_bboxes: positive matched boxes, in the format:
            [NUM_POSITIVES * [xmin, y_min, x_max, y_max]]
        pred_bboxes: positive predicted boxes, in the format:
            [NUM_POSITIVES * [xmin, y_min, x_max, y_max]]
        """ 
        # 1. Compute square of the diagonal of the small enclosing box
        top_left = tf.math.minimum(pred_bboxes[..., :2], gt_bboxes[..., :2])
        bottom_right = tf.math.maximum(pred_bboxes[..., 2:], gt_bboxes[..., 2:])
        width = tf.clip_by_value(bottom_right[..., 0] - top_left[..., 0], 0., 512.)
        height = tf.clip_by_value(bottom_right[..., 1] - top_left[..., 1], 0., 512.)
        c_square = width ** 2 + height ** 2

        # 2. Compute square of the Euclidean distance between the centers of the bboxes
        centered_gt_bboxes = center_bbox(gt_bboxes)
        centered_pred_bboxes = center_bbox(pred_bboxes)
        gt_centers = centered_gt_bboxes[..., :2]
        pred_centers = centered_pred_bboxes[..., :2]
        x1 = gt_centers[..., 0]
        y1 = gt_centers[..., 1]
        x2 = pred_centers[..., 0]
        y2 = pred_centers[..., 1] 
        p_square = (x2 - x1)**2 + (y2 - y1)**2

        # 3. Compute RDIoU
        return p_square / c_square
    
    def av(self, iou, gt_bboxes, pred_bboxes):
        """
        Compute the penalty term for the CIoU loss, based on aspect ratio

        Parameters 
        ----------
        iou: IoU value between gt_bboxes and pred_bboxes
        gt_bboxes: positive matched boxes, in the format:
            [NUM_POSITIVES * [xmin, y_min, x_max, y_max]]
        pred_bboxes: positive predicted boxes, in the format:
            [NUM_POSITIVES * [xmin, y_min, x_max, y_max]]
        """
        # 1. Compute v
        w_gt = gt_bboxes[..., 2] - gt_bboxes[..., 0]
        h_gt = gt_bboxes[..., 3] - gt_bboxes[..., 1]
        w_pred = pred_bboxes[..., 2] - pred_bboxes[..., 0]
        h_pred = pred_bboxes[..., 3] - pred_bboxes[..., 1]
        v = (4/(math.pi**2)) * ((tf.math.atan(w_gt/h_gt) - tf.math.atan(w_pred/h_pred)) ** 2)

        # 2. Compute a
        a = v / (tf.ones_like(iou) - iou + v)

        return a*v
    
    def __call__(self, gt_bboxes, pred_bboxes):
        """
        Apply the desired IoU regression loss with a sum reduction

        Parameters
        ----------
        gt_bboxes: positive matched boxes, in the format:
            [NUM_POSITIVES * [xmin, y_min, x_max, y_max]]
        pred_bboxes: positive predicted boxes, in the format:
            [NUM_POSITIVES * [xmin, y_min, x_max, y_max]]
        """
        loss = None
        iou = jaccard_overlap(pred_bboxes, gt_bboxes, expand=False)
        rdiou = self.RDIoU(pred_bboxes, gt_bboxes)
        diou_loss = tf.ones_like(iou) - iou + rdiou
        if self.loss_type == 'DIoU':
            loss = diou_loss
        elif self.loss_type == 'CIoU':
            av = self.av(iou, gt_bboxes, pred_bboxes)
            loss = diou_loss + av
        else:
            warnings.warn("Wrong loss type. Available 'DIoU' or 'CIoU'")
        return tf.math.reduce_sum(loss)


class SSD_Loss:
    """
    Class for the SSD loss computation
    """
    def __init__(
        self,
        default_boxes,
        num_classes, 
        regression_type='smooth_l1', 
        hard_negative_ratio=3, 
        alpha=ALPHA
    ):
        """
        SSD loss class constructor

        Parameters
        ----------
        default_boxes: default boxes necessary to decode bboxes in the iou loss
        num_classes: the number of classes in the dataset
        regression_type: regression type to compute the localization loss
        hard_negative_ratio: ratio between negative and positive bboxes
        alpha: multiplicative coefficient for the localization loss
        """
        self.default_boxes = default_boxes
        self.num_classes = num_classes
        self.regression_type = regression_type
        self.hard_negative_ratio = hard_negative_ratio
        self.alpha = alpha
        if regression_type == 'DIoU' or 'CIoU':
            self.iou_loss = IoU_Loss(regression_type)
        elif regression_type == 'smooth_l1':
            self.iou_loss = None
        else:
            raise ValueError("Wrong regression type. Available: 'smooth_l1', 'CIoU' or 'DIoU'")

    def hard_negative_mining(self, gt_classes, pred_classes):
        """
        Function to reduce the number of false positives 

        Parameters
        ----------
        gt_classes: ground truth classes
        pred_classes: predicted classes

        Return
        ------
        positive_mask: boolean mask for positive examples
        negative_mask: boolean mask for negative examples 
        """
        confidence_loss = \
            SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        l_conf_tmp = confidence_loss(
            gt_classes, 
            pred_classes
        )
        positive_mask = gt_classes > 0
        positives_num = tf.reduce_sum(tf.cast(positive_mask, tf.int32), axis=1)                 
        negatives_num = self.hard_negative_ratio * positives_num
        sorted_indices = tf.argsort(l_conf_tmp, axis=1, direction='DESCENDING')                            
        negative_mask = tf.argsort(sorted_indices, axis=1) < tf.expand_dims(negatives_num, 1)
        return positive_mask, negative_mask

    def classification_loss(self, gt_classes, pred_classes, positive_mask, negative_mask):
        """
        Apply cross entropy to the classes in order to compute the classification loss

        Parameters
        ----------
        gt_classes: ground truth classes
        pred_classes: predicted classes
        positive_mask: boolean mask for positive examples
        negative_mask: boolean mask for negative examples

        Return
        ------
        l_conf: confidence loss
        """
        confidence_loss = \
            SparseCategoricalCrossentropy(from_logits=True, reduction='sum')
        l_conf = confidence_loss(
            gt_classes[tf.math.logical_or(positive_mask, negative_mask)],
            pred_classes[tf.math.logical_or(positive_mask, negative_mask)]
        )
        return l_conf

    def regression_loss(self, gt_bboxes, pred_bboxes, positive_mask):
        """
        Apply regression function to the classes in order to compute the localization loss

        Parameters
        ----------
        gt_bboxes: ground truth bboxes
        pred_bboxes: predicted bboxes
        positive_mask: boolean mask for positive examples

        Return
        ------
        l_loc: localization loss
        """
        if self.regression_type == 'smooth_l1':
            localization_loss = \
                Huber(delta=1.0, reduction='sum')
            l_loc = localization_loss(
                gt_bboxes[positive_mask], 
                pred_bboxes[positive_mask]
            )
        elif self.regression_type == 'DIoU' or self.regression_type == 'CIoU':
            pred_bboxes = decode_boxes(self.default_boxes, pred_bboxes)[positive_mask]
            gt_bboxes = decode_boxes(self.default_boxes, gt_bboxes)[positive_mask]
            l_loc = self.iou_loss(gt_bboxes, pred_bboxes)
        return l_loc

    def __call__(self, gt_bboxes, gt_classes, prediction):
        """
        Compute the Multibox loss by summing confidence loss and localization loss.
        The result is then divided by the number of positives

        Parameters
        ----------
        gt_bboxes: matched default boxes, in the format:
            [BATCH_SIZE * NUM_DEF_BOXES * [x1_enc, y1_enc, x2_enc, y2_enc]]
        gt_classes: ground truth classes
        prediction: prediction of the ssd, in the format:
            [BATCH_SIZE * TOT_PRED * [x1, y1, x2, y2, l]]

        Return
        ------
        multibox_loss: the multibox loss
        localization_loss: loss obtained with bboxes regression
        confidence_loss: loss obtained with classes confidence
        """
        pred_classes = prediction[..., :self.num_classes]
        pred_bboxes = prediction[..., self.num_classes:]

        # Hard Negative Mining
        positive_mask, negative_mask = \
            self.hard_negative_mining(gt_classes, pred_classes)

        # Confidence Loss
        confidence_loss = \
            self.classification_loss(gt_classes, pred_classes, 
                                     positive_mask, negative_mask)

        # Localization Loss
        localization_loss = \
            self.regression_loss(gt_bboxes, pred_bboxes, 
                                 positive_mask)
            
        # Multibox Loss
        N = tf.reduce_sum(tf.cast(positive_mask, tf.float32))
        if N == 0:
            multibox_loss = N
        else:
            multibox_loss = (confidence_loss + self.alpha * localization_loss) / N   

        return multibox_loss, localization_loss / N, confidence_loss / N
