#!/usr/bin/env python3

from detection_tools import *
from skimage import io
import numpy as np
from collections import namedtuple
from cv2 import resize

from itertools import product
import math


TRAIN_ANN_PATH = './annotations/instances_train2017.json'
VAL_ANN_PATH = './annotations/instances_val2017.json'


class Feature_Map(object):

    def __init__(self, feature_map, aspect_ratios, sk, sk1):
        self.feature_map = feature_map
        self.aspect_ratios = aspect_ratios
        self.sk = sk
        self.sk1 = sk1

    @property
    def shape(self):
        return self.feature_map.shape

    @property
    def default_boxes(self):
        boxes = []
        x_size = self.feature_map.shape[2]
        y_size = self.feature_map.shape[1]
        fk = x_size = y_size
        for i, j in product(range(x_size), range(y_size)):          
            for ar in self.aspect_ratios:
                width = self.sk * math.sqrt(ar)
                height = self.sk / math.sqrt(ar)
                x_center = (i + 0.5) / fk
                y_center = (j + 0.5) / fk
                boxes.append(Bounding_Box(x_center, y_center, width, height))
                if ar == 1:
                    new_sk = math.sqrt(self.sk*self.sk1)
                    width = new_sk * math.sqrt(ar)
                    height = new_sk / math.sqrt(ar)
                    x_center = (i + 0.5) / fk
                    y_center = (j + 0.5) / fk
                    boxes.append(Bounding_Box(x_center, y_center, width, height))
        return boxes


class Dataloader:

    def __init__(
        self, 
        dataset, 
        batch_size=32, 
        image_dim=(300,300), 
        image_source="url", 
        train_obj=None, 
        val_obj=None
    ):  
        self.batch_size = batch_size
        self.image_dim = image_dim
        self.image_source = image_source
        self.dataset = dataset

        if dataset == "COCO":
            self.dataset_name = "COCO"
            if train_obj is not None and val_obj is not None:
                self.train_coco = train_obj
                self.val_coco = val_obj
            else:
                self.train_coco = COCO(TRAIN_ANN_PATH)
                self.val_coco = COCO(VAL_ANN_PATH)
            self.train_ids, self.train_urls, self.labels = global_info(self.train_coco)
            self.val_ids, self.val_urls, self.labels = global_info(self.val_coco)
            self.labels_dict = {i:value for i, value in zip(self.val_coco.getCatIds(), self.labels)}
            self.augmentation = Transform(image_dim=self.image_dim, image_format='coco')
        else:       
            raise TypeError("Wrong or unsupported dataset." +
                            "[available: 'COCO']") 

        # general informations
        print("_____________________________ INFO _____________________________\n")
        print("Train set = %i images [%s]" % (len(self.train_ids), self.image_source))
        print("Eval set = %i images [%s]" % (len(self.val_ids), self.image_source))
        print("%d Labels:\n%s" % (len(self.labels_dict), self.labels_dict))
        print("________________________________________________________________\n") 
        #pause = input("\n\nPress Enter to continue")  

    def preprocess(self, images, ids):
        """
        Apply augmentation

        Parameters
        ----------
        images: list of images to convert
        ids: list of images id

        Return
        ------
        batch: list of dict, with fields:
            'image': image read from the dataset
            'labels': bboxes with label (Labeled_Box object) relative to the image
        """
        batch = []
        for i in range(len(images)):
            if ids[i] in self.train_ids:
                content = get_image_content(self.train_coco, ids[i])
            elif ids[i] in self.val_ids:
                content = get_image_content(self.val_coco, ids[i])
            else:
                raise ValueError("No image in the dataset with id = ", ids[i])
            bboxes, class_labels = [], []
            for c in content:
                if isinstance(c.bbox[0], list):
                    for box in c.bbox:
                        bboxes.append(box)
                        class_labels.append(c.object)
                else:
                    bboxes.append(c.bbox)
                    class_labels.append(c.object)
            transformed = self.augmentation.transform(
                image=images[i],
                bboxes=bboxes,
                class_labels=class_labels
            )
            img_transformed = transformed['image']
            bboxes_transformed = transformed['bboxes']
            labeled_boxes = []
            for box, label in zip(bboxes_transformed, class_labels): 
                labeled_boxes.append(Labeled_Box(bbox=Bounding_Box(*box), label=label))
            batch.append({
                'image': img_transformed,
                'labels': labeled_boxes
            })
        return batch

    def generate_batch(self):
        """
        Generate a train and a validation batch

        Return
        ------
        *_batch: list of dict, with fields:
            'image': image read from the dataset
            'labels': bboxes with label (Labeled_Box object)
        """
        for index in range(0, len(self.train_ids), self.batch_size):
            train_indices = np.random.randint(0, len(self.train_ids), self.batch_size)
            train_batch, val_batch = [], []
            if self.image_source == "url":
                train_urls = [self.train_urls[i] for i in train_indices]
                train_ids = [self.train_ids[i] for i in train_indices]
                val_urls = self.val_urls[index : (index + self.batch_size)]
                val_ids = self.val_ids[index : (index + self.batch_size)]
                print("Reading data from urls...")
                train_imgs = [io.imread(url)/255. for url in train_urls]
                val_imgs = [io.imread(url)/255. for url in val_urls]
                print("Preprocessing images...")
                train_batch = self.preprocess(images=train_imgs, ids=train_ids)
                val_batch = self.preprocess(images=val_imgs, ids=val_ids)
                print("Done!")
            yield train_batch, val_batch
