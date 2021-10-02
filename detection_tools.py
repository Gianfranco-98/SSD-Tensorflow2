#!/usr/bin/env python


# Stock libraries
import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from math import ceil
import tensorflow as tf
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from collections import namedtuple


# _________________________________ Useful structures _________________________________ #


"""
Structure of an element or a group of elements of an image

Fields
------
    object: name of the class in the dataset
    count: # of occurrences of that class in the image
    area: area(s) of the element
    bbox: bbox(es) of the element
"""
Image_Element = namedtuple("Image_Element", field_names = \
        ['object', 'count', 'area', 'bbox'])

"""
Structure of a single bounding box in a feature map

Fields
------
    x1: x_center or x_min
    y1: y_center or y_min
    x2: width    or x_max
    y2: height   or y_max
"""
Bounding_Box = namedtuple("Bounding_Box", field_names = \
    ['x1', 'y1', 'x2', 'y2'])

"""
Structure of a bounding box with class index

Fields
------
    bbox: Bounding_Box structure
    label: class index in the dataset
"""
Labeled_Box = namedtuple("Labeled_Box", field_names = \
    ['bbox', 'label'])


# ____________________________________ Math tools ____________________________________ #


def match_boxes(gt_labels, def_boxes, threshold=0.5): 
    """
    Return the new ground truth, corresponding to labeled default boxes

    Parameters
    ----------
    gt_labels: Tensor of real bounding boxes of the image(s), with labels, in the format:
               [NUM_GT_BOXES * [x_min, y_min, x_max, y_max | label]]
               
    def_boxes: Tensor of default boxes of an image, in the format:
               [NUM_DEF_BOXES * [x_min, y_min, x_max, y_max]]
    threshold: positive threshold

    Return
    ------
    ground_truth: labeled default boxes
    """
    # Handle missing labels                                                     
    if len(gt_labels) == 0:    
        def_labels = tf.zeros(shape=(len(def_boxes),), dtype=tf.float32)
        return def_boxes, def_labels

    # Process ground truth
    gt_boxes = gt_labels[..., :-1]
    labels = gt_labels[..., -1]

    # Compute IoU values
    jaccard = jaccard_overlap(def_boxes, gt_boxes)
    max_gt_jaccard = tf.reduce_max(jaccard, axis=1)
    max_def_jaccard = tf.reduce_max(jaccard, axis=0)
    max_gt_indices = tf.argmax(jaccard, axis=1)
    max_def_indices = tf.argmax(jaccard, axis=0)

    # Ensure best IoU
    max_gt_jaccard = tf.tensor_scatter_nd_update(
        max_gt_jaccard,
        tf.expand_dims(max_def_indices, 1),
        tf.ones_like(max_def_indices, dtype=tf.float32))

    # Match boxes with IoU > threshold
    positive_mask = tf.where(
        max_gt_jaccard >= threshold,
        1.,
        0.
    )
    matched_labels = tf.gather(labels, max_gt_indices) * positive_mask
    matched_boxes = tf.gather(gt_boxes, max_gt_indices)
    encoded_boxes = encode_boxes(def_boxes, matched_boxes)

    return encoded_boxes, matched_labels


def encode_boxes(def_boxes, matched_boxes, variances=[0.1, 0.2]):
    """
    Encode the boxes with centroid encoding and divide by std_dev:
        1 -> get offsets
        2 -> encode variances

    Parameters
    ----------
    matched_boxes: Tensor of matched gt_boxes of all priors:
                   [NUM_PRIORS * [x_min, y_min, x_max, y_max]]
    def_boxes: Tensor of default boxes of an image, in the format:
               [NUM_DEF_BOXES * [x_min, y_min, x_max, y_max]]
    variances: array of [center_variance, width_height_variance]

    Return 
    ------
    default boxes encoded
    """
    # Center bboxes
    def_boxes = center_bbox(def_boxes)
    matched_boxes = center_bbox(matched_boxes)

    # Get center offsets and width/height ratio
    c_offset = (matched_boxes[..., :2] - def_boxes[:, :2])
    we_ratio = (matched_boxes[..., 2:] / def_boxes[:, 2:])

    # Encode variances
    c_offset /= (def_boxes[:, 2:] * variances[0])
    we_ratio = tf.math.log(we_ratio) / variances[1]

    return tf.concat([c_offset, we_ratio], axis=-1)


def decode_boxes(def_boxes, pred_boxes, variances=[0.1, 0.2]):
    """
    Decode the boxes obtained from the ssd prediction:
        1 -> decode variances
        2 -> get coordinates

    Parameters
    ----------
    pred_boxes: Tensor of bboxes predicted by ssd:
                   [BATCH_SIZE * NUM_PRIORS * [x_min, y_min, x_max, y_max]]
    def_boxes: Tensor of default boxes of an image, in the format:
               [NUM_DEF_BOXES * [x_min, y_min, x_max, y_max]]
    variances: array of [center_variance, width_height_variance]

    Return 
    ------
    bboxes: bounding boxes predicted by ssd in top-left format
    """
    # Decode variances
    def_boxes = center_bbox(def_boxes)
    c_offset = pred_boxes[..., :2] * variances[0] * def_boxes[:, 2:]
    we_ratio = tf.math.exp(pred_boxes[..., 2:] * variances[1])

    # Get center coordinates
    c_coords = c_offset + def_boxes[:, :2]
    we_coords = we_ratio * def_boxes[:, 2:]

    # Corner bboxes
    bboxes = corner_bbox(tf.concat([c_coords, we_coords], axis=-1))

    return bboxes


def jaccard_overlap(bbox1, bbox2, expand=True):
    """
    Calculate the IoU between 2 given set of bounding boxes.
    The 'expand' parameter is useful for efficiency reasons when calculating IoU losses.

    Parameters
    ----------
    bbox1: Tensor of default boxes or predicted boxes of an image, in the format:
               [NUM_DEF(NUM_PRED) *  [x_min, y_min, x_max, y_max]]
           If expand = False, the format should be:
               [NUM_DEF(NUM_PRED) *  [[x_min, y_min, x_max, y_max]]]
    bbox2: Tensor of coordinates of the ground truth boxes:
               [NUM_OBJECTS * [x_min, y_min, x_max, y_max]]
           If expand = False, the format should be:
               [[NUM_OBJECTS * [x_min, y_min, x_max, y_max]]]

    Return
    ------
    Jaccard similarity coefficient, in the format:
              Tensor -> shape (NUM_OBJECT, NUM_PRIORS) 
    """
    # Adapt dimensions
    if expand:
        bbox1 = tf.expand_dims(bbox1, 1)
        bbox2 = tf.expand_dims(bbox2, 0)

    intersection = get_intersection(bbox1, bbox2)
    union = get_union(bbox1, bbox2, intersection)
    assert (intersection <= union).numpy().all()

    return intersection / union


def get_intersection(bbox1, bbox2, origin='top_left'):                                  
    """
    Calculate area of the intersection of the 2 given set of bounding boxes

    Parameters
    ----------
    bbox*: Tensor of coordinates of the correspondent boxes:
           NUM_BOXES * [x_min, y_min, x_max, y_max]
    origin: origin of the image coordinate system

    Return
    ------
    Area of the intersection
    """
    top_left = tf.math.maximum(bbox1[..., :2], bbox2[..., :2])
    bottom_right = tf.math.minimum(bbox1[..., 2:], bbox2[..., 2:])

    width = tf.clip_by_value(bottom_right[..., 0] - top_left[..., 0], 0., 512.)
    height = tf.clip_by_value(bottom_right[..., 1] - top_left[..., 1], 0., 512.)

    return width * height


def get_union(bbox1, bbox2, intersection=None):
    """
    Calculate area of the union of the 2 given bounding boxes

    Parameters
    ----------
    box*: Tensor of coordinates of the correspondent boxes:
          NUM_B0XES * [x_min, y_min, x_max, y_max]
    intersection: area of the intersection of the boxes

    Return
    ------
    Area of the union 
    """
    area1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    area2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])
    return (area1 + area2 - intersection)


def center_bbox(bbox):
    """
    Convert bounding box(es) from the format:
        [x_min, y_min, x_max, y_max]
    To:
        [x_center, y_center, width, height]
    """
    xy_center = (bbox[..., :2] + bbox[..., 2:]) / 2
    width_height = (bbox[..., 2:] - bbox[..., :2]) 
    return tf.concat([xy_center, width_height], axis=-1)


def corner_bbox(bbox):                                                          
    """
    Convert a bounding box from the format:
        [x_center, y_center, width, height]
    To: 
        [x_min, y_min, x_max, y_max]
    """
    top_left = bbox[..., :2] - bbox[..., 2:]/2
    bottom_right = bbox[..., :2] + bbox[..., 2:]/2
    return tf.concat([top_left, bottom_right], axis=-1)


def normalize_bbox(bbox, image_width, image_height, label=None, image_format="min_max"):
    """
    1. Corner bbox
        Convert a ground truth bbox to the format:
            [x_min, y_min, x_max, y_max]
    2. Normalize in the range (0, 1)
    3. Add label if not None
    """
    if image_format == "coco":
        bbox = [bbox[0], bbox[1], (bbox[0]+bbox[2]), (bbox[1]+bbox[3])]
    if label is not None:
        gtbox = tf.constant([
            bbox[0] / image_width,
            bbox[1] / image_height,
            bbox[2] / image_width,
            bbox[3] / image_height,
            label
        ], dtype=tf.float32)
    else:
        gtbox = tf.constant([
            bbox[0] / image_width,
            bbox[1] / image_height,
            bbox[2] / image_width,
            bbox[3] / image_height
        ], tf.float32)
    return gtbox

# ___________________________________ General tools ___________________________________ #


def add_bboxes(image, bboxes, classes=None, scores=None, bboxes_format="min_max"):
    """
    Add bounding boxes to an image

    Parameters
    ----------
    image: image to modify
    bboxes: list of bounding boxes to draw on the image
    classes: labels within bounding boxes (no classes printed if None)
    scores: values of accuracy for each bounding box (no scores printed if None)
    bboxes_format: can be 'coco', 'centered', 'min_max'

    Return
    ------
    image: the same image as before, but with the boxes inside
    """

    # TODO: MAKE THE CODE MORE EFFICIENT

    num_boxes = len(bboxes)
    if np.max(bboxes) <= 1.5:                                                   # TMP
        bboxes = np.clip(bboxes, 0, 1)
        bboxes[..., 0] = bboxes[..., 0] * image.shape[1]
        bboxes[..., 1] = bboxes[..., 1] * image.shape[0]
        bboxes[..., 2] = bboxes[..., 2] * image.shape[1]
        bboxes[..., 3] = bboxes[..., 3] * image.shape[0]
    bboxes = np.array(bboxes, dtype=np.int32)
    if np.max(image) < 1.5:
        image = ((image + 1.) * 127.).astype(np.int32)
    for i in range(num_boxes):
        if np.max(image) > 1:
            COLOR = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            COLOR = (random.random(), random.random(), random.random())
        if bboxes_format == "coco":
            bboxes[i] = (int(bboxes[i][0]), int(bboxes[i][1]), 
                         int(bboxes[i][0]+bboxes[i][2]), int(bboxes[i][1]+bboxes[i][3]))
        elif bboxes_format == "centered":
            bboxes[i] = (int(bboxes[i][0]-bboxes[i][2]/2), int(bboxes[i][1]-bboxes[i][3]/2),
                         int(bboxes[i][0]+bboxes[i][2]/2), int(bboxes[i][1]+bboxes[i][3]/2))
        elif bboxes_format == "min_max":
            bboxes[i] = (int(bboxes[i][0]), int(bboxes[i][1]),
                         int(bboxes[i][2]), int(bboxes[i][3]))
        else:
            raise ValueError("Wrong value for 'bboxes_format' arg")
        cv2.rectangle(img=image, pt1=(bboxes[i][0], bboxes[i][1]), pt2=(bboxes[i][2], bboxes[i][3]), 
                      color=COLOR, thickness=1)
        if classes is not None:
            FONT = cv2.FONT_HERSHEY_SIMPLEX
            FONTSCALE = 0.4
            THICKNESS = 1
            text = classes[i]
            if scores is not None:
                text += ' ' + str(scores[i])[:4]
            textsize = cv2.getTextSize(text, FONT, FONTSCALE, THICKNESS)[0]
            cv2.rectangle(img=image, pt1=(bboxes[i][0], bboxes[i][1]), pt2=(bboxes[i][0]+ceil(1.1*textsize[0]), bboxes[i][1]+ceil(1.2*textsize[1])), 
                          color=COLOR, thickness=-1)
            cv2.putText(img=image, text=text, org=(bboxes[i][0], bboxes[i][1]+textsize[1]), 
                        fontFace=FONT, fontScale=FONTSCALE, 
                        color=(255, 255, 255), thickness=THICKNESS)
    return image


def test_bboxes(image, bboxes, bboxes_format='min_max', classnames=None, classnames_dict=None, scores=None):
    """
    Plot bounding boxes and relative classnames on the passed image

    Parameters
    ----------
    image: the image to test
    bboxes: the bboxes to plot
    bboxes_format: 'coco', 'centered' or 'min_max' format
    classnames: labels relative to bboxes
    classnames_dict: dictionary with entries like {class_id -> class_name}
    scores: tensor of predicted classes scores
    """
    if bboxes.shape[0] == 0:
        raise ValueError("bboxes vector is void!")
    if classnames_dict is not None and classnames is not None:
        tmp_arr = np.array(classnames)
        classnames = [classnames_dict[tmp_arr[i]] for i in range(len(tmp_arr))]
    new_image = add_bboxes(np.copy(image), bboxes, classnames, scores, bboxes_format=bboxes_format)
    plt.figure(figsize=(8, 8))
    plt.imshow(new_image)
    plt.show()
    

def lists_from_content(image_content):
    """
    Given a list of Image_Element structures (content of an image), 
    return two lists with all bboxes and their relative classes

    Parameters
    ----------
    image_content: list of Image_Element structures

    Returns
    -------
    bboxes_list: list of all bboxes of the image
    classes_list: list of all the classes for each bbox of the image
    """
    bboxes_list, classes_list = [], []
    for elem in image_content:
        if isinstance(elem.bbox[0], list):
            for box in elem.bbox:
                classes_list.append(elem.object)
                bboxes_list.append(box)
        else:
            bboxes_list.append(elem.bbox)
            classes_list.append(elem.object)
    return bboxes_list, classes_list

# ____________________________________ VOC tools ____________________________________ #


def load_annotations(directory):
    """
    Given VOC annotations directory, load all annotations
    
    Parameters
    ----------
    dir: directory of xml annotation files

    Return
    ------
    roots: list of all annotation roots
    """
    annotations_files = os.listdir(directory)
    annotations_files.sort()
    print("\nLoading annotations into memory...")
    roots = [ET.parse(os.path.join(directory, ann_file)).getroot() 
             for ann_file in tqdm(annotations_files)]
    return roots

# ____________________________________ COCO tools ____________________________________ #


def get_image_content(coco, ID):
    """
    Given COCO object and image ID, return content of the image.
    For instance, if "img" is an img with 2 dogs and 1 cat:
        content = [
            2 dog with area = [[...],[...]] and bbox = [[...],[...]]
            1 cat with area = ... and bbox = [...]
        ]
    
    Parameters
    ----------
    coco: COCO object, containg information of all images
    ID: ID of the single image to examine

    Return
    ------
    content: list of Image_Element objects
    """
    content = []
    tmp_list = []
    for ann_dict in coco.anns.items():
        if ann_dict[1]['image_id'] == ID:
            if ann_dict[1]['category_id'] not in tmp_list:
                tmp_list.append(ann_dict[1]['category_id'])
                element = {
                    'object': ann_dict[1]['category_id'], 
                    'count': 1, 
                    'area': ann_dict[1]['area'], 
                    'bbox': ann_dict[1]['bbox']}
                content.append(element)
            else:
                for i in range(len(tmp_list)):
                    if ann_dict[1]['category_id'] == tmp_list[i]:
                        content[i]['count'] += 1
                        if content[i]['count'] > 2:
                            content[i]['area'].append(ann_dict[1]['area'])
                            content[i]['bbox'].append(ann_dict[1]['bbox'])
                        else:
                            content[i]['area'] = [content[i]['area'], ann_dict[1]['area']]
                            content[i]['bbox'] = [content[i]['bbox'], ann_dict[1]['bbox']]
    return [Image_Element(**elem) for elem in content]