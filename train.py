#!/usr/bin/env python

# __________________________________________ Libraries __________________________________________ #


# Dataset
from pycocotools.coco import COCO

# Networks
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

# Math
import numpy as np

# Generic
from tensorboardX import SummaryWriter
import time
import os

# My files
from ssd import SSD
from loss import SSD_Loss
from configuration import *
from detection_tools import *
from image_detection import *
from train_utilities import *
from data.dataset import COCO_Dataset, VOC_Dataset

# ____________________________________________ Main ____________________________________________ #


if __name__ == "__main__":

    print("\n\n\n_______ Welcome to SSD Multibox training _______\n")
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
    print("\n\tTraining on %s dataset" % (DATASET_NAME + DATASET_KEY))
    dataset.show_info()
    _ = input("Press Enter to continue...")

    ## 2. Dataloader initialization
    print("\t2. Dataloader initialization...")
    dataloader = Dataloader(
        dataset, 
        BATCH_SIZE
    )
    train_generator = dataloader.generate_batch("train")

    ## 3. Network initialization
    print("\t3. Network initialization...")
    ssd = SSD(num_classes=len(dataset.label_ids)+1, input_shape=INPUT_SHAPE)
    checkpoint = tf.train.Checkpoint(ssd)
    ssd.summary()
    _ = input("Press Enter to continue...")   

    ## 4. Generate default boxes
    print("\t4. Default boxes generation...")
    fm_shapes = ssd.output_shape
    aspect_ratios = ASPECT_RATIOS
    scales = SCALES
    default_boxes = Image.generate_default_boxes(fm_shapes, aspect_ratios, scales)     

    ## 5. Learning initializations
    print("\t5. Learning initialization...")
    learning_rate = PiecewiseConstantDecay(
        boundaries = BOUNDARIES,
        values = LR_VALUES
    )
    ssd_optimizer = SGD(
        learning_rate = learning_rate,
        momentum = MOMENTUM
    )
    ssd_loss = SSD_Loss(
        default_boxes = default_boxes,
        num_classes = ssd.num_classes, 
        regression_type = REGRESSION_TYPE, 
        hard_negative_ratio = 3, 
        alpha = ALPHA
    )           

    ## 6. Training initializations
    print("\t6. Final training initializations...")
    last_iter = 0
    iterations = []
    mb_losses, loc_losses, conf_losses = [], [], []
    if TENSORBOARD_LOGS:
        writer = SummaryWriter(comment = "SSD | __" + DATASET_NAME + DATASET_KEY + "__")
    if LOAD_MODEL:
        print("Loading latest train data...")
        ssd, iterations, mb_losses, loc_losses, conf_losses = \
            load_train_data(ssd, CHECKPOINT_DIR)
        last_iter = iterations[-1]
        if TENSORBOARD_LOGS:
            for i in range(last_iter):
                writer.add_scalar("Multibox loss", mb_losses[i], i)
                writer.add_scalar("Confidence loss", conf_losses[i], i)
                writer.add_scalar("Localization loss", loc_losses[i], i)
    # ---------------------------------------------------------------- #
    
    print("Initialization completed!")
    print("Start Training...")

    # -------------------------- Train loop -------------------------- #
    for iteration in range(last_iter+1, ITERATIONS):

        # Load data
        glob_start = time.time()
        try:
            train_imgs, train_labels, train_ids = next(train_generator)
        except StopIteration:
            train_generator = dataloader.generate_batch("train")
            train_imgs, train_labels, train_ids = next(train_generator)
        batch_size = len(train_imgs)

        # Match bounding boxes
        matched_boxes, def_labels = [], []
        for b in range(batch_size):
            boxes, labels = match_boxes(train_labels[b], default_boxes)
            matched_boxes.append(boxes)
            def_labels.append(labels)

        # Predict and learn
        input_imgs = np.stack(train_imgs, axis=0)
        matched_boxes = tf.stack(matched_boxes, axis=0)
        def_labels = tf.stack(def_labels, axis=0)
        multibox_loss, localization_loss, confidence_loss = \
            learn(ssd, ssd_optimizer, ssd_loss, input_imgs, matched_boxes, def_labels)
        print("[%d] (%f s)   -   Multibox loss = |%f|, Localization loss = |%f|, Confidence_loss = |%f|" % 
            (iteration, time.time() - glob_start, multibox_loss, localization_loss, confidence_loss))
        
        # Plot train process
        iterations.append(iteration)
        mb_losses.append(multibox_loss)
        loc_losses.append(localization_loss)
        conf_losses.append(confidence_loss)
        if iteration % PLOT_PERIOD == 0 and iteration > 0:
            plot_train_data(iterations, mb_losses, loc_losses, conf_losses)

        # Update Tensorboard Writer
        if TENSORBOARD_LOGS:
            writer.add_scalar("Multibox loss", multibox_loss.numpy(), iteration)
            writer.add_scalar("Confidence loss", confidence_loss.numpy(), iteration)
            writer.add_scalar("Localization loss", localization_loss.numpy(), iteration)

        # Save checkpoint
        if iteration % CHECKPOINT_PERIOD == 0 and iteration > 0:
            print(" - Saving Train data...")
            save_train_data(checkpoint, CHECKPOINT_FILEPATH, CHECKPOINT_DIR, iterations, mb_losses, loc_losses, conf_losses)
