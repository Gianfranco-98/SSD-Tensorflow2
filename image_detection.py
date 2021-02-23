#!/usr/bin/env python3

from COCO_utils import *
from augmentation import *
from skimage import io
import numpy as np
from collections import namedtuple

from itertools import product

import math


TRAIN_ANN_PATH = './annotations/instances_train2017.json'
VAL_ANN_PATH = './annotations/instances_val2017.json'

# Structure of an element or a group of elements of an image
Image_Element = namedtuple("Image_Element", field_names = \
        ['object', 'count', 'area', 'bbox'])


# Structure of a single default box in a feature map
Default_Box = namedtuple("Default_Box", field_names = \
    ['x_center, y_center, width, height'])


class Image:

    def __init__(self, image, width, height, url=None):
        self.image = image
        self.width = width
        self.height = height
        self.url = url


class COCO_Image(Image):

    def __init__(self, image, width, height, url=None, ID=None, content=None):
        super().__init__(image, width, height, url)
        self.ID = ID
        self.content = [Image_Element(**elem) for elem in content]


class Feature_Map(object):

    def __init__(self, feature_map, aspect_ratios, sk, sk1):
        self.feature_map = feature_map
        self.aspect_ratios = aspect_ratios
        self.sk = sk
        self.sk1 = sk1

    @property
    def default_boxes(self):
        boxes = []
        x_size = self.feature_map.shape[2]
        y_size = self.feature_map.shape[1]
        fk = x_size = y_size
        for i, j in product(range(x_size), range(y_size)):          #TODO: Debug iterations
            for ar in self.aspect_ratios:
                width = self.sk * math.sqrt(ar)
                height = self.sk / math.sqrt(ar)
                x_center = (i + 0.5) / fk
                y_center = (j + 0.5) / fk
                boxes.append(Default_Box(x_center, y_center, width, height))
                if ar == 1:
                    new_sk = math.sqrt(self.sk*self.sk1)
                    width = new_sk * math.sqrt(ar)
                    height = new_sk / math.sqrt(ar)
                    x_center = (i + 0.5) / fk
                    y_center = (j + 0.5) / fk
                    boxes.append(Default_Box(x_center, y_center, width, height))
        return boxes


class Dataloader:

    def __init__(self, dataset, batch_size=32, image_sorce="url"):  
        self.batch_size = batch_size
        self.image_sorce = image_sorce

        if dataset == "COCO":
            self.train_coco = COCO(TRAIN_ANN_PATH)
            self.val_coco = COCO(VAL_ANN_PATH)
            train_ids, train_urls, labels = global_info(self.train_coco)
            val_ids, val_urls, labels = global_info(self.val_coco)
            labels_dict = {i:value for i, value in zip(self.val_coco.getCatIds(), labels)}
            if image_sorce == "url":
                self.X_train = train_urls
                self.X_val = val_urls
                self.labels = labels
                self.labels_dict = labels_dict
        else:       
            raise TypeError("Wrong or unsupported dataset." +
                            "[available: 'COCO']") 

        # general informations
        print("_____________________________ INFO _____________________________\n")
        print("Train set = %i images [%s]" % (len(self.X_train), self.image_sorce))
        print("Eval set = %i images [%s]" % (len(self.X_val), self.image_sorce))
        print("Labels:\n", labels)
        print("________________________________________________________________\n") 
        pause = input("\n\nPress Enter to continue")  

    def preprocess(self, batch):
        pass

    def generate_batch(self):
        for index in range(0, len(self.X_val), self.batch_size):
            train_indices = np.random.randint(0, len(self.X_train), self.batch_size)
            train_batch = [self.X_train[i] for i in train_indices]
            #train_batch = self.preprocess(train_batch)
            val_batch = self.X_val[index : (index + self.batch_size)]
            if self.image_sorce == "url":
                print("Reading urls...")
                train_batch = [io.imread(url) for url in train_batch]
                val_batch = [io.imread(url) for url in val_batch]
                print("Done!")
            yield train_batch, val_batch