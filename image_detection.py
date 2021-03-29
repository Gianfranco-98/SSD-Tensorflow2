#!/usr/bin/env python3

# Dataset 
from data_configuration import *

# Image
from detection_tools import *
from skimage import io
from cv2 import resize

# Math
import matplotlib.pyplot as plt
import numpy as np
import math

# Generic
from collections import namedtuple
from itertools import product
import multiprocessing as mp


class Image(object):

    def __init__(self, image=None, name=None):
        self.image = image
        self.name = name

    @staticmethod
    def generate_default_boxes(feature_map_shapes, aspect_ratios, scales):
        default_boxes = []
        for i in range(len(scales)-1):
            if feature_map_shapes[i][0] is None:
                feature_map_shapes[i] = (1, feature_map_shapes[i][1], feature_map_shapes[i][2])
            fm = Feature_Map(np.zeros(feature_map_shapes[i]), aspect_ratios[i], scales[i], scales[i+1])
            for box in fm.default_boxes:
                default_boxes.append(corner_bbox(tf.constant(box)))
        return default_boxes
      

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
        batch_size=32 
    ):  
        self.data = dataset
        self.batch_size = batch_size
        if self.data.name == "COCO":
            self.image_format = "coco"
        elif self.data.name == "VOC":
            self.image_format = "pascal_voc"
        else:
            raise ValueError("Wrong or unsupported dataset. Available: 'COCO' or 'VOC'")
        self.augmentation = Transform(image_dim=self.data.image_dim, image_format=self.image_format)

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
            pool = mp.Pool(processes=NUM_WORKERS)
            if phase == "train":
                indices = np.random.randint(0, len(self.data.train_ids), self.batch_size)
                ids = [self.data.train_ids[i] for i in indices]
                if self.data.image_source == "url":
                    print(" - Reading data from urls...")
                    urls = [self.data.train_urls[i] for i in indices]
                    imgs = pool.map(self.read_url, urls)
                else:
                    print(" - Reading data from disk...")
                    filenames = [self.data.train_filenames[i] for i in indices]
                    imgs = pool.map(self.read_img, filenames)
            else:
                indices = list(range(index, (index + self.batch_size)))
                ids = [self.data.val_ids[i] for i in indices]
                if self.data.image_source == "url":
                    print(" - Reading data from urls...")
                    urls = [self.data.val_urls[i] for i in indices]
                    imgs = pool.map(self.read_url, urls)
                else:
                    print(" - Reading data from disk...")
                    filenames = [self.data.val_filenames[i] for i in indices]
                    imgs = pool.map(self.read_img, indices)          
            pool.close()
            pool.join()                    
            print(" - Preprocessing images...")
            images, labels = self.preprocess(imgs, ids, phase)
            yield images, labels, ids

    def preprocess(self, images, ids, phase, normalize=True):                                          
        """
        Apply augmentation to image and bounding boxes and
        convert to convenient structures

        Parameters
        ----------
        images: list of images to convert
        ids: list of images id
        phase: 'train' or 'eval'
        normalize: if True, bboxes are normalized in range [0, 1]
                  and converted to list format (useful during train
                  for efficiency reasons)

        Return
        ------
        batch: list of dict, with fields:
            'image': image read from the dataset
            'labels': bboxes with label (Labeled_Box object) relative to the image
        """
        processed_images, processed_labels = [], []
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
            class_labels = [self.data.labels_dict[cl] for cl in class_labels]
            labeled_boxes = []
            if normalize:
                labeled_boxes = [normalize_bbox(
                                  bboxes_transformed[i],
                                  self.data.image_dim[0], self.data.image_dim[1],
                                  class_labels[i],
                                  self.image_format)
                                 for i in range(len(class_labels))]
            else:
                for box, label in zip(bboxes_transformed, class_labels): 
                    labeled_boxes.append(Labeled_Box(bbox=Bounding_Box(*box), label=label))
            processed_images.append(img_transformed)
            processed_labels.append(labeled_boxes)
        return processed_images, processed_labels

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

    @staticmethod
    def read_img(filename):
        """
        Read an image, given the filename
        """
        return cv2.imread(filename)

    @staticmethod
    def read_url(url):
        """
        Read an URL into an image
        """
        return io.imread(url)
