#!/usr/bin/env python3

from detection_tools import *
from skimage import io
import numpy as np
from collections import namedtuple
from cv2 import resize
import matplotlib.pyplot as plt

from itertools import product
import math
import time


TRAIN_ANN_PATH = './annotations/instances_train2017.json'
VAL_ANN_PATH = './annotations/instances_val2017.json'


class Feature_Map(object):

    def __init__(self, feature_map, aspect_ratios, sk, sk1, clip_box=True):
        self.feature_map = feature_map
        self.aspect_ratios = aspect_ratios
        self.sk = sk
        self.sk1 = sk1
        self.clip = clip_box

    @property
    def shape(self):
        return self.feature_map.shape

    @property
    def default_boxes(self):
        boxes = []
        x_size = self.shape[2]
        y_size = self.shape[1]
        fk = x_size = y_size
        for i, j in product(range(x_size), range(y_size)):          
            for ar in self.aspect_ratios:
                width = self.sk * math.sqrt(ar)
                height = self.sk / math.sqrt(ar)
                x_center = (j + 0.5) / fk
                y_center = (i + 0.5) / fk
                boxes.append([x_center, y_center, width, height])
                if ar == 1:
                    new_sk = math.sqrt(self.sk*self.sk1)
                    x_center = (i + 0.5) / fk
                    y_center = (j + 0.5) / fk
                    boxes.append([x_center, y_center, new_sk, new_sk])
        
        # clip and convert to Bounding_Box format
        for b, box in enumerate(boxes):
            if self.clip:
                boxes[b] = np.clip(box, a_min=0, a_max=1)
            boxes[b] = Bounding_Box(*box)
        return boxes


class Dataset:

    def __init__(
        self,
        train_obj,
        val_obj,
        name="COCO",
        image_dim=(300, 300),
        n_channels=3,
        image_source="url"
    ):
        self.name = name
        self.image_dim = image_dim
        self.n_channels = n_channels
        self.image_source = image_source

        if self.name == "COCO":
            self.train_coco = train_obj
            self.val_coco = val_obj
            self.train_ids, self.train_urls, self.labels = global_info(self.train_coco)
            self.val_ids, self.val_urls, self.labels = global_info(self.val_coco)
            self.train_bboxes, self.train_labels = get_detection_data(self.train_coco, self.train_ids)
            self.val_bboxes, self.val_labels = get_detection_data(self.val_coco, self.val_ids)
            self.labels_names = {i:value for i, value in zip(self.val_coco.getCatIds(), self.labels)}
            self.labels_dict = {i:j for i,j in zip(self.val_coco.getCatIds(), list(range(1, len(self.labels)+1)))}
        else:
            raise ValueError("Wrong name for 'name' arg. Available 'COCO'")

    def get_info(self):
        print("_____________________________ INFO _____________________________\n")
        print("Train set = %i images [%s]" % (len(self.train_ids), self.image_source))
        print("Eval set = %i images [%s]" % (len(self.val_ids), self.image_source))
        print("%d Labels:\n%s" % (len(self.labels_names), self.labels_names))
        print("________________________________________________________________\n") 


class Dataloader:

    def __init__(
        self, 
        dataset, 
        batch_size=32 
    ):  
        self.data = dataset
        self.batch_size = batch_size

        if self.data.name == "COCO":
            self.augmentation = Transform(image_dim=self.data.image_dim, image_format='coco')
        else:       
            raise TypeError("Wrong or unsupported dataset. Available: 'COCO'")

    def generate_batch(self, phase="train"):
        """
        Generate a train and a validation batch

        Parameters
        ----------
        phase: 'train' or 'eval'

        Return
        ------
        batch: list of dict, with fields:
            'image': image read from the dataset
            'labels': bboxes with label (Labeled_Box object)
        ids: ids of the images in the batch
        """
        ids, urls = [], []
        if phase == "train":
            generator_len = len(self.data.train_ids)
        elif phase == "eval":
            generator_len = len(self.data.val_ids)
        else:
            raise ValueError("Wrong value for arg 'phase'. Available 'train' or 'eval'")
        
        for index in range(0, generator_len, self.batch_size):
            if phase == "train":
                indices = np.random.randint(0, len(self.data.train_ids), self.batch_size)
                ids = [self.data.train_ids[i] for i in indices]
                if self.data.image_source == "url":
                    print(" - Reading data from urls...")
                    start = time.time()
                    urls = [self.data.train_urls[i] for i in indices]
                    imgs = [io.imread(url) for url in urls]
                    end = time.time()
                    print("\t - [%f s]" % (end - start))
                else:
                    #TODO: manage loading from disk
                    pass
            else:
                ids = self.data.val_ids[index : (index + self.batch_size)]
                if self.data.image_source == "url":
                    print(" - Reading data from urls...")
                    start = time.time()
                    urls = self.data.val_urls[index : (index + self.batch_size)]
                    imgs = [io.imread(url) for url in urls]
                    end = time.time()
                    print("\t - [%f s]" % (end - start))
                else:
                    #TODO: manage loading from disk
                    pass
            print(" - Preprocessing images...")
            start = time.time()
            batch = self.preprocess(imgs, ids, phase)
            end = time.time()
            print("\t - [%f s]" % (end - start))
            yield batch, ids

    def preprocess(self, images, ids, phase):                                          
        """
        Apply augmentation to image and bounding boxes and
        convert to convenient structures

        Parameters
        ----------
        images: list of images to convert
        ids: list of images id
        phase: 'train' or 'eval'

        Return
        ------
        batch: list of dict, with fields:
            'image': image read from the dataset
            'labels': bboxes with label (Labeled_Box object) relative to the image
        """
        batch = []
        for i in range(len(images)):
            img_index = self.get_img_index(ids[i], phase)
            if phase == "train":
                bboxes = self.data.train_bboxes[img_index]
                class_labels = self.data.train_labels[img_index]
            else:
                bboxes = self.data.val_bboxes[img_index]
                class_labels = self.data.val_labels[img_index]                
            if len(images[i].shape) == 2 and self.data.n_channels == 3:
                images[i] = self.gray_to_rgb(images[i])
            transformed = self.augmentation.transform(
                image=images[i],
                bboxes=bboxes,
                class_labels=class_labels
            )
            img_transformed = transformed['image']
            bboxes_transformed = transformed['bboxes']
            class_labels = [self.data.labels_dict[class_labels[i]] for i in range(len(class_labels))]
            labeled_boxes = []
            for box, label in zip(bboxes_transformed, class_labels): 
                labeled_boxes.append(Labeled_Box(bbox=Bounding_Box(*box), label=label))
            batch.append({
                'image': img_transformed,
                'labels': labeled_boxes
            })
        return batch

    def get_img_index(self, id, phase):
        """
        Return the index of the image in the dataset
        """
        index = self.data.train_ids.index(id) if phase == "train" else self.data.val_ids.index(id)
        return index

    @staticmethod
    def gray_to_rgb(image):
        """
        Convert a grayscale image to RGB, by adding channels
        """
        dim = np.zeros(shape=(image.shape[0], image.shape[1]))
        red_image = np.stack((image/255., dim, dim), axis=2)
        green_image = np.stack((dim, image/255., dim), axis=2)
        blu_image = np.stack((dim, dim, image/255.), axis=2)
        rgb_image = red_image + green_image + blu_image
        return rgb_image
