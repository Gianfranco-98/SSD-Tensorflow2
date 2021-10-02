#!/usr/bin/env python


# Stock libraries
import random
import numpy as np
import tensorflow as tf
import albumentations as A
from math import floor, ceil

# My libraries
from configuration import *
from detection_tools import *


class Transform:
    """
    Transformation class
    """
    def __init__(
        self, 
        image_dim, 
        image_format='coco', 
        iou_thresholdes=IOU_THRESHOLDES, 
        min_patch_ratio=0.1, 
        max_patch_ratio=1
    ):
        """
        Transformation class constructor
        Available transformations:
            - Resize
            - Horizontal flip
            - Color jitter
            - Image patching 

        Parameters
        ----------
        image_dim: desired image dimension after the resize op
        image_format: format of the bounding boxes to adjust
        iou_thresholdes: various thresholdes for the patching algorithm
        min_patch_ratio: minimum ratio of the patch
        max_patch_ratio: maximum ratio of the patch
        """
        self.dim = image_dim
        self.format = image_format
        self.iou_thresholdes = iou_thresholdes
        self.min_patch_ratio = min_patch_ratio
        self.max_patch_ratio = max_patch_ratio

    def basic_augmentation(self, image, bboxes, class_labels, phase):
        """
        Apply basic augmentations to the image:
          - Resize to a fixed size
          - Random horizontal flip
          - ColorJitter (random photometric distortions)

        Parameters
        ----------
        image: The image to augment
        bboxes: Tensor of bounding boxes in the image
        class_labels: list of labels relative to the objects in the image
        phase: 'train', 'eval' or 'test'

        Return
        ------
        The image transformed with new bounding boxes and labels
        """
        transform = None
        if phase == 'train':
            transform = A.Compose([                 
                # Geometric modifications
                A.Resize(height=self.dim[0], width=self.dim[1], always_apply=True),
                A.HorizontalFlip(p=0.5),
                # Photometric modifications
                A.ColorJitter()
            ], bbox_params=A.BboxParams(format=self.format, label_fields=['class_labels']))
        else:
            transform = A.Compose([                 
                A.Resize(height=self.dim[0], width=self.dim[1], always_apply=True),
            ], bbox_params=A.BboxParams(format=self.format, label_fields=['class_labels']))

        return transform(image=image, bboxes=bboxes, class_labels=class_labels)

    def sample_patch(self, image, gt_boxes, labels, patchfind_attemps=PATCHFIND_ATTEMPTS):
        """
        Sample a patch from the image randomly. Three options:
          - Original image
          - Random patch
          - Patch such that the IoU with the objects is > threshold (randomly selected)

        Parameters
        ----------
        image: The CV2 image to patch  
        gt_boxes: Tensor of bounding boxes in the image
        labels: List of labels relative to the objects in the image
        patchfind_attemps: Attempts to find a valid patch for the image

        Return
        ------
        image: The image patched
        valid_boxes: Tensor of valid bounding boxes contained in the patch
        valid_labels: list of labels relative to the objects in the patch
        """
        choice = np.random.choice(['original_image', 'random_patch'], p=[0.9, 0.1])
        if choice == 'original_image':    # Maintain original image size
            return image, gt_boxes, labels
        elif choice == 'random_patch':    # Select a patch in function of the IoU
            threshold = np.random.choice(self.iou_thresholdes)
            valid_boxes, valid_labels = None, None
            patch = None
            overlaps = None

            # Patch sampling algorithm
            for i in range(PATCHFIND_ATTEMPTS):

                # 1. Randomly generate Patch
                patch_width = random.uniform(self.min_patch_ratio, self.max_patch_ratio) * image.shape[1]
                patch_height = random.uniform(self.min_patch_ratio, self.max_patch_ratio) * image.shape[0]
                patch_xmin = random.uniform(0, image.shape[1]-patch_width)
                patch_ymin = random.uniform(0, image.shape[0]-patch_height)
                patch = np.array(
                    [patch_xmin, patch_ymin, 
                     patch_xmin + patch_width, patch_ymin + patch_height], 
                    dtype=np.float32)
                
                # 2. Compute jaccard overlaps
                overlaps = jaccard_overlap(tf.expand_dims(patch, 0), gt_boxes)

                # 3. Check for bboxes with a jaccard greater than the threshold
                if not tf.math.reduce_any(overlaps >= threshold):
                    continue

                # 4. Check if the patch contains a bbox center             
                gt_boxes_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
                valid_indices = ((overlaps > threshold) &
                    (gt_boxes_centers[:, 0] > patch[0]) & (gt_boxes_centers[:, 0] < patch[2]) &
                    (gt_boxes_centers[:, 1] > patch[1]) & (gt_boxes_centers[:, 1] < patch[3]))
                if not tf.math.reduce_any(valid_indices):
                    continue

                # 5. Adjust bboxes and image
                valid_boxes = gt_boxes[tf.squeeze(valid_indices, 0)]
                valid_labels = (np.array(labels)[valid_indices]).tolist()
                valid_boxes = tf.stack([
                    tf.math.maximum(valid_boxes[:, 0], patch[0]) - patch[0],
                    tf.math.maximum(valid_boxes[:, 1], patch[1]) - patch[1],
                    tf.math.minimum(valid_boxes[:, 2], patch[2]) - patch[0],
                    tf.math.minimum(valid_boxes[:, 3], patch[3]) - patch[1]], axis=1)
                image = image[floor(patch[1]):ceil(patch[3]), floor(patch[0]):ceil(patch[2])]
                break

            if valid_boxes is None:
                valid_boxes = gt_boxes
                valid_labels = labels

            return image, valid_boxes, valid_labels

    def __call__(self, image, gt_boxes, labels, phase):
        """
        Apply augmentations to the image

        Parameters
        ----------
        image: The CV2 image to augment  
        gt_boxes: Tensor of bounding boxes in the image
        labels: list of labels relative to the objects in the image
        phase: 'train', 'eval' or 'test'

        Return
        ------
        Image transformed, adjusted bounding boxes and remaining labels
        """
        if phase != 'test':
            image, gt_boxes, labels = \
                self.sample_patch(image, gt_boxes, labels)
        transformed = \
            self.basic_augmentation(image, gt_boxes, labels, phase)
        return transformed['image'], transformed['bboxes'], labels