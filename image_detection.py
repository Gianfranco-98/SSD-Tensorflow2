#!/usr/bin/env python3

from detection_tools import *
from skimage import io
import numpy as np
from collections import namedtuple
from cv2 import resize
import matplotlib.pyplot as plt

from itertools import product
import math

from tensorflow.image import draw_bounding_boxes


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


class Dataloader:

    def __init__(
        self, 
        dataset, 
        batch_size=32, 
        image_dim=(300,300), 
        n_channels=3,
        image_source="url", 
        train_obj=None, 
        val_obj=None
    ):  
        self.batch_size = batch_size
        self.image_dim = image_dim
        self.n_channels = n_channels
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
            self.labels_names = {i:value for i, value in zip(self.val_coco.getCatIds(), self.labels)}
            self.labels_dict = {i:j for i,j in zip(self.val_coco.getCatIds(), list(range(1, len(self.labels)+1)))}
            self.augmentation = Transform(image_dim=self.image_dim, image_format='coco')
        else:       
            raise TypeError("Wrong or unsupported dataset." +
                            "[available: 'COCO']") 

        # general informations
        print("_____________________________ INFO _____________________________\n")
        print("Train set = %i images [%s]" % (len(self.train_ids), self.image_source))
        print("Eval set = %i images [%s]" % (len(self.val_ids), self.image_source))
        print("%d Labels:\n%s" % (len(self.labels_names), self.labels_names))
        print("________________________________________________________________\n") 
        #pause = input("\n\nPress Enter to continue")  

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
            generator_len = len(self.train_ids)
        elif phase == "eval":
            generator_len = len(self.val_ids)
        else:
            raise ValueError("Wrong value for arg 'phase'. Available 'train' or 'eval'")
        
        for index in range(0, generator_len, self.batch_size):
            if phase == "train":
                indices = np.random.randint(0, len(self.train_ids), self.batch_size)
                ids = [self.train_ids[i] for i in indices]
                if self.image_source == "url":
                    urls = [self.train_urls[i] for i in indices]
                else:
                    #TODO: manage loading from disk
                    pass
            else:
                ids = self.val_ids[index : (index + self.batch_size)]
                if self.image_source == "url":
                    urls = self.val_urls[index : (index + self.batch_size)]
                else:
                    #TODO: manage loading from disk
                    pass
            if len(urls) > 0:
                print(" - Reading data from urls...")
                imgs = [io.imread(url) for url in urls]
            print(" - Preprocessing images...")
            batch = self.preprocess(imgs, ids)
            yield batch, ids

    def preprocess(self, images, ids):                                          #TODO: speed up
        """
        Apply augmentation to image and bounding boxes and
        convert to convenient structures

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
            bboxes, class_labels = lists_from_content(content)
            if len(images[i].shape) == 2 and self.n_channels == 3:
                images[i] = self.gray_to_rgb(images[i])
            transformed = self.augmentation.transform(
                image=images[i],
                bboxes=bboxes,
                class_labels=class_labels
            )
            img_transformed = transformed['image']
            bboxes_transformed = transformed['bboxes']
            class_labels = [self.labels_dict[class_labels[i]] for i in range(len(class_labels))]
            labeled_boxes = []
            for box, label in zip(bboxes_transformed, class_labels): 
                labeled_boxes.append(Labeled_Box(bbox=Bounding_Box(*box), label=label))
            batch.append({
                'image': img_transformed,
                'labels': labeled_boxes
            })
        return batch

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


class Dataloader:

    def __init__(
        self, 
        dataset, 
        batch_size=32, 
        image_dim=(300,300), 
        n_channels=3,
        image_source="url", 
        train_obj=None, 
        val_obj=None
    ):  
        self.batch_size = batch_size
        self.image_dim = image_dim
        self.n_channels = n_channels
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
            self.labels_names = {i:value for i, value in zip(self.val_coco.getCatIds(), self.labels)}
            self.labels_dict = {i:j for i,j in zip(self.val_coco.getCatIds(), list(range(1, len(self.labels)+1)))}
            self.augmentation = Transform(image_dim=self.image_dim, image_format='coco')
        else:       
            raise TypeError("Wrong or unsupported dataset." +
                            "[available: 'COCO']") 

        # general informations
        print("_____________________________ INFO _____________________________\n")
        print("Train set = %i images [%s]" % (len(self.train_ids), self.image_source))
        print("Eval set = %i images [%s]" % (len(self.val_ids), self.image_source))
        print("%d Labels:\n%s" % (len(self.labels_names), self.labels_names))
        print("________________________________________________________________\n") 
        #pause = input("\n\nPress Enter to continue")  

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
            generator_len = len(self.train_ids)
        elif phase == "eval":
            generator_len = len(self.val_ids)
        else:
            raise ValueError("Wrong value for arg 'phase'. Available 'train' or 'eval'")
        
        for index in range(0, generator_len, self.batch_size):
            if phase == "train":
                indices = np.random.randint(0, len(self.train_ids), self.batch_size)
                for i in indices:
                    ids.append(self.train_ids[i])
                    if self.image_source == "url":
                        urls.append(self.train_urls[i])
            else:
                for i in indices:
                    ids.append(self.val_ids[i])
                    if self.image_source == "url":
                        urls.append(self.val_urls[i])
            if len(urls) > 0:
                print(" - Reading data from urls...")
                imgs = [io.imread(url) for url in urls]
            else:
                #TODO: manage loading from disk
                pass
            print(" - Preprocessing images...")
            batch = self.preprocess(imgs, ids)
            yield batch, ids

    def preprocess(self, images, ids):                                          #TODO: speed up
        """
        Apply augmentation to image and bounding boxes and
        convert to convenient structures

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
            bboxes, class_labels = lists_from_content(content)
            if len(images[i].shape) == 2 and self.n_channels == 3:
                images[i] = self.gray_to_rgb(images[i])
            transformed = self.augmentation.transform(
                image=images[i],
                bboxes=bboxes,
                class_labels=class_labels
            )
            img_transformed = transformed['image']
            bboxes_transformed = transformed['bboxes']
            class_labels = [self.labels_dict[class_labels[i]] for i in range(len(class_labels))]
            labeled_boxes = []
            for box, label in zip(bboxes_transformed, class_labels): 
                labeled_boxes.append(Labeled_Box(bbox=Bounding_Box(*box), label=label))
            batch.append({
                'image': img_transformed,
                'labels': labeled_boxes
            })
        return batch

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
