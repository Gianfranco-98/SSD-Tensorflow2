#!/usr/bin/env python


# Stock libraries
import tensorflow as tf
import numpy as np
import warnings
import time

# My libraries
from ssd import SSD
from configuration import *
from detection_tools import *
from image_detection import *
from data.dataset import COCO_Dataset, VOC_Dataset


def inference(
        ssd, 
        images, 
        default_boxes, 
        conf_threshold=CONFIDENCE_THRESHOLD, 
        loc_threshold=JACCARD_THRESHOLD,
        num_nms_output=MAX_NMS_BOXES,
        top_k=TOP_K_BOXES
    ):
    """
    Predict boxes, labels and scores and apply nms + top_k

    Parameters
    ----------
    images: batch of images
    default_boxes: default boxes useful to decode predicted bboxes
    conf_treshold: threshold to select only reasonable confidence scores
    loc_threshold: threshold for the nms
    num_nms_output: maximum number of boxes in output from the nms
    top_k: maximum number of boxes in output from this function

    Return
    ------
    final_boxes: predicted bounding boxes
    final_labels: predicted labels
    final_scores: predicted scores
    """

    # Prediction
    feature_maps = ssd(images)
    ssd_prediction = ssd.process_feature_maps(feature_maps)
    pred_classes = ssd_prediction[..., :ssd.num_classes]
    pred_classes = tf.squeeze(pred_classes, axis=0)
    pred_classes = tf.math.softmax(pred_classes, axis=-1)
    pred_bboxes = ssd_prediction[..., ssd.num_classes:]
    pred_bboxes = tf.squeeze(pred_bboxes, axis=0)

    # Bounding Boxes Decoding
    bboxes = decode_boxes(default_boxes, pred_bboxes)

    # Filtering
    final_scores, final_boxes, final_labels = [], [], []
    for label in range(1, ssd.num_classes):

        ## Confidence Thresholding
        class_scores = pred_classes[:, label]
        valid_scores = class_scores[class_scores >= conf_threshold]        
        valid_boxes = bboxes[class_scores >= conf_threshold]

        ## Non-Maximum Suppression
        """ 
        WARNING: tf.image.non_max_suppression needs [y_min, x_min, y_max, x_max] 
                 bboxes format, but after some tests we found that it leads to 
                 the same results with the format [x_min, y_min, x_max, y_max].
        xmin = tf.expand_dims(valid_boxes[..., -4], axis=-1)
        ymin = tf.expand_dims(valid_boxes[..., -3], axis=-1)
        xmax = tf.expand_dims(valid_boxes[..., -2], axis=-1)
        ymax = tf.expand_dims(valid_boxes[..., -1], axis=-1)
        valid_boxes = tf.concat([ymin, xmin, ymax, xmax], -1)
        """
        nms_indices = tf.image.non_max_suppression(
            valid_boxes, valid_scores, num_nms_output, loc_threshold)
        maximum_scores = tf.gather(valid_scores, nms_indices)
        maximum_boxes = tf.gather(valid_boxes, nms_indices)
        class_labels = [label] * maximum_boxes.shape[0]
        final_scores.append(maximum_scores)
        final_boxes.append(maximum_boxes)
        final_labels.append(class_labels)

    final_scores = tf.concat(final_scores, axis=0)
    final_boxes = tf.concat(final_boxes, axis=0)
    final_labels = np.concatenate(final_labels)

    # Top-K Filtering   
    if len(final_scores) > top_k:
        final_scores, final_indices = tf.math.top_k(final_scores, k=top_k)
        final_boxes = tf.gather(final_boxes, final_indices)
        final_boxes = tf.clip_by_value(final_boxes, 0., 1.)
        final_labels = np.take(final_labels, final_indices)
    else:
        warnings.warn("No predicted scores for the selected parameters. Try to reduce 'conf_threshold' first.")

    return final_boxes.numpy(), final_labels, final_scores.numpy()


if __name__ == "__main__":

    print("\n\n\n_______ Welcome to SSD Multibox testing _______\n")
    print("Initialization...")

    # ------------------------ Initialization ------------------------ #
    ## 1. Dataset initialization
    print("\t1. Dataset inizialization...")
    if DATASET_NAME == "COCO":
        train_coco = COCO(TRAIN_ANN_PATH)
        val_coco = COCO(VAL_ANN_PATH)
        test_coco = COCO(TEST_ANN_PATH)
        dataset = COCO_Dataset(
            train_coco,
            val_coco,
            test_coco
        )
    elif DATASET_NAME == "VOC":
        train_roots = load_annotations(TRAIN_ANN_PATH)
        val_roots = load_annotations(VAL_ANN_PATH)
        test_roots = load_annotations(TEST_ANN_PATH)
        dataset = VOC_Dataset(
            train_roots,
            val_roots,
            test_roots
        )
    else:
        raise ValueError("Wrong or unsupported dataset. Available 'COCO' or 'VOC'")
    print("\n\Testing on %s dataset" % (DATASET_NAME + TESTSET_YEAR))
    dataset.show_info()
    _ = input("Press Enter to continue...")

    ## 2. Dataloader initialization
    print("\t2. Dataloader initialization...")
    dataloader = Dataloader(
        dataset, 
        TEST_SIZE
    )
    test_generator = dataloader.generate_batch("test")

    ## 3. Network initialization
    print("\t3. Network initialization...")
    ssd = SSD(num_classes=len(dataset.label_ids)+1, input_shape=INPUT_SHAPE)
    latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    ssd.load_weights(latest)
    ssd.summary()
    _ = input("Press Enter to continue...")   

    ## 4. Generate default boxes
    print("\t4. Default boxes generation...")
    fm_shapes = ssd.output_shape
    aspect_ratios = ASPECT_RATIOS
    scales = SCALES
    default_boxes = Image.generate_default_boxes(fm_shapes, aspect_ratios, scales)          
    # ---------------------------------------------------------------- #
    
    print("Initialization completed!")
    print("Start Testing...")

    # -------------------------- Test loop -------------------------- #
    for iteration in range(TEST_ITERATIONS):

        # Load data
        print("\n________Test iteration %d________" % iteration)
        print("1.1 Data loading")
        glob_start = time.time()
        try:
            test_imgs, test_labels, test_ids = next(test_generator)
        except StopIteration:
            test_generator = dataloader.generate_batch("test")
            test_imgs, test_labels, test_ids = next(test_generator)
        batch_size = len(test_imgs)

        # Inference
        print("2. Inference")
        infer_time = time.time()
        input_imgs = np.stack(test_imgs, 0)
        vb, l, scores,  = inference(ssd, np.expand_dims(input_imgs[0], 0), default_boxes, 0.4)
        print("Inference time =", time.time() - infer_time)
        
        # Show ground truth image boxes
        gt_boxes = np.stack(test_labels[0], axis=0)[..., :-1]
        gt_labels = np.stack(test_labels[0], axis=0)[..., -1]
        test_bboxes(test_imgs[0], gt_boxes, 'min_max', gt_labels, dataset.classnames_dict)

        # Show predicted image boxes
        if vb.shape[0] != 0:
            test_bboxes(test_imgs[0], vb, 'min_max', l, dataset.classnames_dict, scores)
        else:
            print("No bboxes predicted")
        
        print("___Done in %f s!___" % (time.time() - glob_start))