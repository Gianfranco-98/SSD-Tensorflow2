#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

ALPHA = 1

def ssd_loss(ground_truth, prediction, num_classes=80, hard_negative_ratio=3, alpha=ALPHA):
    """
    Calculate multibox loss for the ssd prediction
    
    Parameters
    ----------
    ground_truth: all default boxes matched with a ground truth box
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
    # Process ground truth
    gt_classes = []
    gt_bboxes = []
    batch_size = len(ground_truth)
    for b in range(batch_size):
        gt_classes.append([gtbox.label for gtbox in ground_truth[b]])
        gt_bboxes.append([list(gtbox.bbox) for gtbox in ground_truth[b]])
    gt_classes = tf.stack(gt_classes)
    gt_bboxes = tf.stack(gt_bboxes)

    # Process predicted data
    pred_classes = prediction[..., :num_classes]
    pred_bboxes = prediction[..., num_classes:]
    print(pred_classes.shape, pred_classes[0][0])
    
    # Hard Negative Mining
    confidence_loss = SparseCategoricalCrossentropy(
        from_logits=False, reduction='none'                       #TODO: check if they are logits
    )
    l_conf = confidence_loss(gt_classes, pred_classes)
    print(l_conf)
    positive_mask = gt_classes > 0
    positives_num = len(positive_mask)
    negatives_num = hard_negative_ratio * positives_num
    sorted_indices = tf.argsort(l_conf, axis=0, direction='DESCENDING')                             #TODO: adjust axis ?
    negative_mask = tf.argsort(sorted_indices, axis=0) < negatives_num      #TODO: expand dims ?
    
    # Confidence loss                                                                                #TODO: if possible, avoid recompute classification loss
    confidence_loss = SparseCategoricalCrossentropy(
        from_logits=False, reduction='sum'                       
    )
    l_conf = confidence_loss(
        gt_classes[tf.math.logical_or(positive_mask, negative_mask)],
        pred_classes[tf.math.logical_or(positive_mask, negative_mask)]
    )

    # Localization loss
    localization_loss = tf.keras.losses.Huber(delta=1.0, reduction='sum')
    l_loc = localization_loss(gt_bboxes[positive_mask], pred_bboxes[positive_mask])

    # multibox loss
    if positives_num == 0:
        loss = 0                                            #TODO: WHY 0 ?
    else:
        loss = (l_conf + alpha * l_loc) / positives_num     #TODO: ensure 1/N is correct here

    return loss, l_loc, l_conf
