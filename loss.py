#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

ALPHA = 1

def ssd_loss(gt_bboxes, gt_classes, prediction, num_classes, hard_negative_ratio=3, alpha=ALPHA):
    """
    Calculate multibox loss for the ssd prediction
    
    Parameters
    ----------
    gt_bboxes: tensor of all default boxes matched with a ground truth box
    gt_classes: tensor of all labels assigned to matched default boxes 
    prediction: ssd prediction
    num_classes: total number of labels in the dataset
    hard_negative_ratio: negative:positive examples ratio
    alpha: coefficient for localization loss

    Return
    ------
    loss: 1/len(matched_boxes) * (confidence_loss + alpha * localization_loss)
    l_loc: error in object localization
    l_conf: error in object classification
    """
    # Process predicted data
    pred_classes = prediction[..., :num_classes]
    pred_bboxes = prediction[..., num_classes:]
    
    # Hard Negative Mining
    confidence_loss = \
        SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    l_conf_tmp = confidence_loss(
        gt_classes, 
        pred_classes
    )
    positive_mask = gt_classes > 0
    positives_num = tf.reduce_sum(tf.cast(positive_mask, tf.int32), axis=1)                 
    negatives_num = hard_negative_ratio * positives_num
    sorted_indices = tf.argsort(l_conf_tmp, axis=1, direction='DESCENDING')                            
    negative_mask = tf.argsort(sorted_indices, axis=1) < tf.expand_dims(negatives_num, 1)
    
    # Confidence loss                                                                                
    confidence_loss = \
        SparseCategoricalCrossentropy(from_logits=True, reduction='sum')
    l_conf = confidence_loss(
        gt_classes[tf.math.logical_or(positive_mask, negative_mask)],
        pred_classes[tf.math.logical_or(positive_mask, negative_mask)]
    )

    # Localization loss
    localization_loss = \
        tf.keras.losses.Huber(delta=1.0, reduction='sum')
    l_loc = localization_loss(
        gt_bboxes[positive_mask], 
        pred_bboxes[positive_mask]
    )

    # multibox loss
    N = tf.reduce_sum(tf.cast(positive_mask, tf.float32))
    if N == 0:
        loss = N
    else:
        loss = (l_conf + alpha * l_loc) / N   

    return loss, l_loc, l_conf
