#!/usr/bin/env python3

# Dataset
from pycocotools.coco import COCO
import xml.etree.ElementTree as ET

# Generic
from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
import time
import os

# My files
from detection_tools import *


# Dataset configuration ------------------------------------------------------ #
DATASET_NAME = "VOC"
DATASET_YEAR = "2012"
DATA_PATH = '/content/data/' + DATASET_NAME + DATASET_YEAR
TRAIN_ANN_PATH = None
VAL_ANN_PATH = None
ANN_PATH = None

if DATASET_NAME == "COCO":
    TRAIN_ANN_PATH = DATA_PATH + '/annotations/instances_train' + DATASET_YEAR + '.json'
    VAL_ANN_PATH = DATA_PATH + '/annotations/instances_val' + DATASET_YEAR + '.json'
    SCALES = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
elif DATASET_NAME == "VOC":
    TRAIN_ANN_PATH = DATA_PATH + '/Annotations'
    SCALES = [0.10, 0.20, 0.37, 0.54, 0.71, 0.88, 1.05]
# ---------------------------------------------------------------------------- #


class Dataset(ABC):

    def __init__(
        self,
        name,
        image_dim=(300, 300),
        n_channels=3,
        image_source="url"
    ):
        if name not in ['COCO', 'VOC']:
            raise ValueError("Wrong dataset name. Available 'COCO' or 'VOC'")
        self.name = name
        self.image_dim = image_dim
        self.n_channels = n_channels
        self.image_source = image_source
        self.train_ids, self.val_ids, self.test_ids = None, None, None
        self.train_urls, self.val_urls, self.test_urls = None, None, None
        self.train_filenames, self.val_filenames, self.test_filenames = None, None, None
        self.classnames_dict, self.labels_dict = None, None
        self.label_ids, self.label_names = None, None
        self.train_bboxes, self.val_bboxes, self.test_bboxes = None, None, None

    def create_dicts(self):
        """
        Create basic dictionaries for the dataset
        """
        self.classnames_dict = dict(zip(self.label_ids, self.label_names))
        self.labels_dict = dict(zip(self.label_ids, range(1, len(self.label_ids)+1)))

    def show_info(self):
        print("_____________________________ INFO _____________________________\n")
        print("Train set = %i images [%s]" % (len(self.train_ids), self.image_source))
        print("Eval set = %i images [%s]" % (len(self.val_ids), self.image_source))
        print("%d Labels:\n%s" % (len(self.classnames_dict), self.classnames_dict))
        print("________________________________________________________________\n") 
    
    @abstractmethod
    def global_info(self, obj):
        """
        Generate dataset global infos

        Parameters
        ----------
        obj: can be a COCO object or a list of VOC xml roots

        Return
        ------
        img_ids: ID of all images in the dataset
        urls: url //
        filenames: name //
        """
        raise NotImplementedError

    @abstractmethod
    def label_info(self, obj):
        """
        Generate dataset label infos

        Parameters
        ----------
        obj: can be a COCO object or a list of VOC xml roots

        Return
        ------
        label_ids: id of all the labels
        label_names: name of each label
        """
        raise NotImplementedError

    @abstractmethod    
    def get_detection_data(self, obj):
        """
        Return the list of ALL bounding box and relative labels for each image

        Parameters
        ----------
        obj: can be a COCO object or a list of VOC xml roots

        Return
        ------
        bboxes: 3-dimensional list: bounding boxes of all images
        labels: 2-dimensional list: labels of all images
        """
        raise NotImplementedError


class COCO_Dataset(Dataset):

    def __init__(
        self,
        train_coco,
        val_coco
    ):
        super(COCO_Dataset, self).__init__(name="COCO", image_source="url")
        self.train_obj = train_coco
        self.val_obj = val_coco

        # get basic info
        self.train_ids, self.train_urls, self.train_filenames = self.global_info(train_coco) 
        self.val_ids, self.val_urls, self.val_filenames = self.global_info(val_coco)
        self.label_ids, self.label_names = self.label_info(train_coco)
        self.create_dicts()

        # get detection data
        self.train_bboxes, self.train_labels = self.get_detection_data(train_coco)
        self.val_bboxes, self.val_labels = self.get_detection_data(val_coco)

    def global_info(self, coco):
        img_ids = coco.getImgIds()
        urls = [coco.imgs[id]['coco_url'] for id in img_ids]
        filenames = []  #TODO: manage COCO in disk
        return img_ids, urls, filenames
    
    def label_info(self, coco):
        label_ids = coco.getCatIds()
        label_names = [coco.cats[id]['name'] for id in label_ids]
        return label_ids, label_names
    
    def get_detection_data(self, coco):
        ID_list = coco.getImgIds()
        bboxes = [[] for i in range(len(ID_list))]
        labels = [[] for i in range(len(ID_list))]
        for ann_dict in tqdm(coco.anns.items()):
            index = ID_list.index(ann_dict[1]['image_id'])
            bboxes[index].append(ann_dict[1]['bbox'])
            labels[index].append(ann_dict[1]['category_id'])
        return bboxes, labels


class VOC_Dataset(Dataset):

    def __init__(
        self,
        train_roots,
        val_roots=None
    ):
        super(VOC_Dataset, self).__init__(name="VOC", image_source="disk")
        self.train_obj = train_roots

        # get basic info
        self.train_ids, self.train_urls, self.train_filenames = self.global_info(train_roots) 
        self.val_ids, self.val_urls, self.val_filenames = self.global_info(val_roots)
        self.label_ids, self.label_names = self.label_info(train_roots)
        self.create_dicts()

        # get detection data
        self.train_bboxes, self.train_labels = self.get_detection_data(train_roots)
        self.val_bboxes, self.val_labels = self.get_detection_data(val_roots)

    def global_info(self, roots):
        ids, urls, filenames = [], [], []
        if roots is not None:
            for root in roots:
                filename = root.find('filename').text
                filenames.append(filename)
                ids.append(filename[:-4])
        return ids, urls, filenames

    def label_info(self, roots):
        label_ids, label_names = [], []
        if roots is not None:
            for root in roots:
                for obj in root.iter('object'):
                    label = obj.find("name").text.lower().strip()
                    if label not in label_names:
                        label_names.append(label)
        label_names.sort()
        label_ids = list(range(1, len(label_names)+1))
        return label_ids, label_names              
    
    def get_detection_data(self, roots):
      if roots is not None:
            format = ['xmin', 'ymin', 'xmax', 'ymax']
            bboxes = [[] for i in range(len(roots))]
            labels = [[] for i in range(len(roots))]
            names_dict = {name:id for id, name in self.classnames_dict.items()}
            for i, root in enumerate(roots):
                for obj in root.iter('object'):
                    bbox_obj = obj.find("bndbox")
                    label_obj = obj.find("name")
                    labels[i].append(names_dict[label_obj.text.lower().strip()])
                    bboxes[i].append([int(bbox_obj.find(coord).text) - 1 for coord in format])
      else:
          bboxes, labels = [], []
      return bboxes, labels
