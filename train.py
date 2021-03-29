#!/usr/bin/env python3

# ___________________________________________________ Libraries ___________________________________________________ #


# Images 
import cv2
from cv2 import imread
from skimage import io
import matplotlib.pyplot as plt
from image_detection import *
from detection_tools import *

# Learning 
import tensorflow as tf
from SSD import SSD

# Math
import numpy as np

# Config
from data_configuration import *

# ___________________________________________________ Constants ___________________________________________________ #


# Files parameters
CHECKPOINT_DIR = './Checkpoints'
CHECKPOINT_FILEPATH = CHECKPOINT_DIR + '/checkpoint'

# Network parameters
BASE_WEIGHTS = 'imagenet'
BASE_NAME = "VGG16"
DEFAULT_BOXES_NUM = [4, 6, 6, 6, 4, 4]
ASPECT_RATIOS = [[1., 2., 1/2],
                 [1., 2., 3., 1/2, 1/3],
                 [1., 2., 3., 1/2, 1/3],
                 [1., 2., 3., 1/2, 1/3],
                 [1., 2., 1/2],
                 [1., 2., 1/2]]
SCALES = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
DEFAULT_BOXES_NUM = [4, 6, 6, 6, 4, 4]
INPUT_SHAPE = (300, 300, 3)
IMAGE_DIM = (300, 300)
N_CHANNELS = 3

# Learning parameters
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

# Train 
CHECKPOINT_PERIOD = 1000
ITERATIONS = 240000
BATCH_SIZE = 32
NUM_WORKERS = 8

# ____________________________________________________ Functions ____________________________________________________ #


def learn(ssd, optimizer, images, matched_boxes, labels):
    with tf.GradientTape() as ssd_tape:
        print(" - Prediction...")
        feature_maps = ssd(images)
        ssd_prediction = ssd.process_feature_maps(feature_maps)
        print(" - Calculating Loss...")
        multibox_loss, localization_loss, confidence_loss = \
            ssd_loss(
                gt_bboxes=matched_boxes,
                gt_classes=labels,
                prediction=ssd_prediction,
                num_classes=ssd.num_classes
            )
    print(" - Optimization...")
    ssd_gradients = ssd_tape.gradient(multibox_loss, ssd.trainable_variables)
    ssd_optimizer.apply_gradients(zip(ssd_gradients, ssd.trainable_variables))
    return multibox_loss, localization_loss, confidence_loss

# ______________________________________________________ Main ______________________________________________________ #


if __name__ == '__main__':
    
    # Dataset initialization
    if DATASET_NAME == "COCO":
        train_coco = COCO(TRAIN_ANN_PATH)
        val_coco = COCO(VAL_ANN_PATH)
        dataset = COCO_Dataset(
            train_coco,
            val_coco,
    )
    elif DATASET_NAME == "VOC":
        roots = load_annotations(TRAIN_ANN_PATH)
        dataset = VOC_Dataset(
            roots
        )
    else:
        raise ValueError("Wrong or unsupported dataset. Available 'COCO' or 'VOC'")
    dataset.show_info()

    dataloader = Dataloader(
        dataset, 
        BATCH_SIZE
    )
    train_generator = dataloader.generate_batch("train")

    # Network initialization
    ssd = SSD(num_classes=len(dataset.labels)+1, input_shape=INPUT_SHAPE)
    ssd_optimizer = SGDW(
        learning_rate = LEARNING_RATE,         #TODO: add learning rate schedule
        weight_decay = WEIGHT_DECAY,
        momentum = MOMENTUM
    )
    checkpoint = tf.train.Checkpoint(ssd)
    ssd.summary()  

    # Generating default boxes
    fm_shapes = ssd.output_shape
    aspect_ratios = ASPECT_RATIOS
    scales = SCALES
    default_boxes = Image.generate_default_boxes(fm_shapes, aspect_ratios, scales)
    default_boxes = tf.stack(default_boxes, axis=0)

    # Train
    for iteration in range(ITERATIONS):

        # Load data
        print("\n________Train iteration %d________" % iteration)
        print("1.1 Data loading")
        glob_start = time.time()
        train_imgs, train_labels, train_ids = next(train_generator)
        batch_size = len(train_imgs)

        # Adapt input to base model
        print("1.2 Additional preprocessing")
        print(" - Adapt images for %s model..." % ssd.base_architecture)
        input_imgs = [preprocess_input(img) for img in train_imgs]

        # Match bounding boxes
        print(" - Matching bboxes...")
        matched_boxes, def_labels = [], []
        for b in range(batch_size):
            tl = tf.cast(tf.stack(train_labels[b], axis=0), dtype=tf.float32)
            boxes, labels = match_boxes(tl, default_boxes)
            matched_boxes.append(boxes)
            def_labels.append(labels)

        # Predict and learn
        print("2. Learning step")
        input_imgs = np.stack(input_imgs, axis=0)
        matched_boxes = tf.stack(matched_boxes, axis=0)
        def_labels = tf.stack(def_labels, axis=0)
        multibox_loss, localization_loss, confidence_loss = \
            learn(ssd, ssd_optimizer, input_imgs, matched_boxes, def_labels)
        print(" Localization loss = ", localization_loss)
        print(" Confidence loss = ", confidence_loss)
        print(" Multibox loss = ", multibox_loss)

        # Save checkpoint
        if iteration % CHECKPOINT_PERIOD == 0 and iteration > 0:
            print(" - Saving Weights...")
            save_path = checkpoint.save(CHECKPOINT_FILEPATH)
        
        print("___Done in %f s!___" % (time.time() - glob_start))