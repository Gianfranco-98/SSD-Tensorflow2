#!/usr/bin/env python3

from pycocotools.coco import COCO
from collections import namedtuple
import cv2

import numpy as np

import albumentations as A

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
    x: x coordinate of the box (can be x_center, x_min, ecc.)
    y: y          //           (can be y_center, y_min, ecc.)
    width: width of the box
    height: height //
"""
Bounding_Box = namedtuple("Bounding_Box", field_names = \
    ['x', 'y', 'width', 'height'])

"""
Structure of a bounding box with class index

Fields
------
    bbox: Bounding_Box structure
    label: class index in the dataset
"""
Labeled_Box = namedtuple("Labeled_Box", field_names = \
    ['bbox', 'label'])

# ___________________________________ Augmentation ___________________________________ #


class Transform:

    def __init__(self, image_dim, image_format='coco'):
        self.format = image_format
        self.dim = image_dim
        self.transform = A.Compose([
            A.Resize(height=self.dim[0], width=self.dim[1], always_apply=True),
            A.HorizontalFlip(p=0.5)
        ], bbox_params=A.BboxParams(format=self.format, label_fields=['class_labels']))    

# ___________________________________ General tools ___________________________________ #


def match_boxes(gt_boxes, pred_boxes, threshold=0.5):
    """
    Match predicted boxes with ground truth boxes, as the following criterion:
        1. match the ones with the best IoU (jaccard overlap)
        2. match all boxes with an IoU > threshold
    
    Parameters
    ----------
    gt_boxes: ground truth boxes (Labeled_Box structures)
    pred_boxes: default boxes generated by predicted feature maps
    threshold: minimum value for IoU to match boxes

    Return
    ------
    ground_truth: the new ground truth (Labeled_Box structures)
    """
    ground_truth = np.zeros(len(pred_boxes))
    global_overlaps = []

    # Match boxes with maximum IoU
    for gtbox in gt_boxes:
        gt_overlaps = [jaccard_overlap(gtbox.bbox, pbox) for pbox in pred_boxes]
        global_overlaps.append(gt_overlaps)
        max_index = overlaps.index(max(gt_overlaps))
        ground_truth[max_index] = \
            encode_box(gtbox, pred_boxes[max_index], positive=True) 

    # Match other boxes with IoU > thresold
    global_overlaps = np.array(global_overlaps)
    max_overlaps = np.max(global_overlaps, axis=0)
    max_overlaps_idxs = np.argmax(global_overlaps, axis=0)
    for def_idx, max_idx in zip(range(len(pred_boxes)), max_overlaps_idxs):
        if max_overlaps[def_idx] > threshold:
            if ground_truth[def_idx] == 0:
                ground_truth[def_idx] = \
                    encode_box(gt_boxes[max_idx], pred_boxes[def_idx], positive=True)
        elif ground_truth[def_idx] == 0:
            ground_truth[def_idx] = \
                encode_box(gt_boxes[max_idx], pred_boxes[def_idx], positive=False)


def encode_box(gt_box, def_box, positive=True, variances=[0.1, 0.1, 0.2, 0.2]):
    """
    Encode the box with centroid encoding and divide by std_dev:
        1 -> get bounding box offsets
        2 -> subtract by variance

    Parameters
    ----------
    gt_box: ground truth bounding boxes (Labeled_Box structure)
    def_box: default boxes predicted by ssd (Bounding_Box structure)
    positive: whether the IoU is under the thresold. If False, label assigned is 0
    variances: array of x_variance, y_variance, width_variance, height_variance

    Return 
    ------
    encoded_box: Labeled_Box structure, with centroid encoding
    """
    x = ((gt_box.bbox.x - def_box.x) / def_box.width) / np.sqrt(variances[0])
    y = ((gt_box.bbox.y - def_box.y) / def_box.height) / np.sqrt(variances[1])
    width = (np.log(gt_box.bbox.width / def_box.width)) / np.sqrt(variances[2])
    height = (np.log(gt_box.bbox.height / def_box.height)) / np.sqrt(variances[3])
    label = gt_box.label if positive else 0
    encoded_box = Labeled_Box(
        bbox=Bounding_Box(x, y, width, height),
        label=label
    )
    return encoded_box


def jaccard_overlap(bbox1, bbox2):
    """
    Calculate the IoU between 2 given bounding boxes

    Parameters
    ----------
    bbox*: coordinates of the correspondent box, in the format:
          Bounding_Box(x, y, width, height), with x=x_center and y=y_center

    Return
    ------
    Jaccard similarity coefficient between two boxes 
    """
    intersection = get_intersection(bbox1, bbox2)
    union = get_union(bbox1, bbox2, intersection)
    return intersection / union


def get_intersection(bbox1, bbox2, origin='top_left'):
    """
    Calculate area of the intersection of the 2 given bounding boxes

    Parameters
    ----------
    bbox*: coordinates of the correspondent bbox, in the format:
          Bounding_Box(x, y, width, height), with x=x_center and y=y_center
    origin: origin of the image coordinate system

    Return
    ------
    Area of the intersection
    """
    if origin == 'top_left':
        left_cx = min(bbox1.x, bbox2.x)
        right_cx = max(bbox1.x, bbox2.x)
        down_cy = max(bbox1.y, bbox2.y)                           
        up_cy = min(bbox1.y, bbox2.y)
        sum_to_left = bbox1.width/2 if left_cx == bbox1.x else bbox2.width/2
        sub_from_right = bbox1.width/2 if right_cx == bbox1.x else bbox2.width/2
        sub_from_down = bbox1.height/2 if down_cy == bbox1.y else bbox2.height/2
        sum_to_up = bbox1.height/2 if up_cy == bbox1.y else bbox2.height/2
        x_intersect = max(0, ((left_cx + sum_to_left) - (right_cx - sub_from_right)))       #TODO: check if the new calculus are correct
        y_intersect = max(0, ((down_cy - sub_from_down) - (up_cy + sum_to_up)))
    else:
        raise ValueError("Wrong value for 'origin' argument ['top_left' is available].")
    return x_intersect * y_intersect


def get_union(bbox1, bbox2, intersection=None):
    """
    Calculate area of the union of the 2 given bounding boxes

    Parameters
    ----------
    box*: coordinates of the correspondent box, in the format:
          Bounding_Box(x, y, width, height), with x=x_center and y=y_center
    intersection: area of the intersection of the boxes

    Return
    ------
    Area of the union 
    """
    area1 = bbox1.width*bbox1.height
    area2 = bbox2.width*bbox2.height
    if intersection is None:
        intersection = get_intersection(bbox1, bbox2)
    return (area1 + area2 - intersection)


def add_bboxes(image, bboxes, classes=None, scores=None, bboxes_format="coco"):
    """
    Add bounding boxes to an image

    Parameters
    ----------
    image: image to modify
    bboxes: list of bounding boxes to draw on the image
    classes: labels within bounding boxes (no classes printed if None)
    scores: values of accuracy for each bounding box (no scores printed if None)

    Return
    ------
    image: the same image as before, but with the boxes inside
    """
    num_boxes = len(bboxes)
    for i in range(num_boxes):
        if bboxes_format == "coco":
            bboxes[i] = (int(bboxes[i][0]), int(bboxes[i][1]), 
                         int(bboxes[i][0]+bboxes[i][2]), int(bboxes[i][1]+bboxes[i][3]))
        elif bboxes_format == "centered_coco":
            bboxes[i] = (int(bboxes[i][0]-bboxes[i][2]/2), int(bboxes[i][1]-bboxes[i][3]/2),
                         int(bboxes[i][0]+bboxes[i][2]/2), int(bboxes[i][1]+bboxes[i][3]/2))
        else:
            raise ValueError("Wrong value for 'bboxes_format' arg")
        cv2.rectangle(img=image, pt1=(bboxes[i][0], bboxes[i][1]), pt2=(bboxes[i][2], bboxes[i][3]), 
                      color=(255, 0, 0), thickness=1)
        cv2.putText(img=image, text=classes[i], org=(bboxes[i][0], bboxes[i][1] - 10), 
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3, 
                    color=(0, 255, 255), thickness=2)
    return image


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


def global_info(coco):
    """
    Obtain generic dataset infos from COCO object

    Parameters
    ----------
    coco: COCO object

    Return
    ------
    ids: ID of all images in the dataset
    urls: url //
    labels: label // 
    """
    urls = []
    ids = coco.getImgIds()
    for i, img_id in enumerate(ids):
        img_meta = coco.imgs[img_id]
        urls.append(img_meta['coco_url'])
    labels = [coco.cats[id]['name'] for id in coco.getCatIds()]
    return ids, urls, labels


def center_coco_bbox(bbox):
    """
    Convert a coco ground truth bbox in the format:
        Bounding_Box(x_min, y_min, width, heigth)
    In order to have it in the format: 
        Bounding_Box(x_center, y_center, width, height)
    """
    x_center = bbox.x + bbox.width/2
    y_center = bbox.y + bbox.height/2
    return Bounding_Box(x_center, y_center, bbox.width, bbox.height)
