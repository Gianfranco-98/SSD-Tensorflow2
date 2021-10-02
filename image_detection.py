#!/usr/bin/env python


# Stock libraries
import os
import cv2
import math
import random
import numpy as np
from skimage import io
import tensorflow as tf
import multiprocessing as mp
from itertools import product

# My libraries
from configuration import *
from detection_tools import *
from data_augmentation import Transform


class Image(object):
    """
    Base class for an image in object detection context
    """
    def __init__(self, image=None, name=None):
        self.image = image
        self.name = name

    @staticmethod
    def generate_default_boxes(feature_map_shapes, aspect_ratios, scales):
        """
        Generate default bounding boxes for all the feature maps

        Parameters
        ----------
        feature_map_shapes: the shapes of the decreasing size feature maps
        aspect_ratios: the aspect ratios of the bboxes relative to all the feature maps
        scales: scales of the bboxes relative to all the feature maps
        """
        default_boxes = None
        for i in range(len(scales)-1):
            if feature_map_shapes[i][0] is None:
                feature_map_shapes[i] = (1, feature_map_shapes[i][1], feature_map_shapes[i][2])
            fm = Feature_Map(np.zeros(feature_map_shapes[i]), aspect_ratios[i], scales[i], scales[i+1])
            df = corner_bbox(tf.clip_by_value(fm.default_boxes, 0., 1.))
            default_boxes = tf.concat([default_boxes, df], 0) if default_boxes is not None else df
        return tf.cast(default_boxes, tf.float32)


class Feature_Map(object):
    """
    Base class for a feature map in object detection context
    """
    def __init__(self, feature_map, aspect_ratios, sk, sk1, clip_box=True):
        """
        Feature map class constructor

        Parameters
        ----------
        feature_map: the feature_map gathered from the network output
        aspect_ratios: the aspect ratios of the bboxes relative to this feature map
        """
        self.feature_map = feature_map
        self.aspect_ratios = aspect_ratios
        self.sk = sk
        self.sk1 = sk1
        self.clip = clip_box

    @property
    def shape(self):
        """
        Shape of the feature map
        """
        return self.feature_map.shape

    @property
    def default_boxes(self):
        """
        Generate default boxes for this feature map
        """
        boxes = []
        x_size = self.shape[2]
        y_size = self.shape[1]
        fk = x_size = y_size
        for i, j in product(range(x_size), range(y_size)):  
            x_center = (j + 0.5) / fk
            y_center = (i + 0.5) / fk     
            for ar in self.aspect_ratios:
                width = self.sk * math.sqrt(ar)
                height = self.sk / math.sqrt(ar)
                boxes.append([x_center, y_center, width, height])
                if ar == 1:
                    new_sk = math.sqrt(self.sk*self.sk1)
                    boxes.append([x_center, y_center, new_sk, new_sk])
        
        # clip and convert to Bounding_Box format
        for b in range(len(boxes)):
            if self.clip:
                boxes[b] = np.clip(boxes[b], 0, 1)
        return tf.stack(boxes, 0)


class Dataloader:
    """
    Dataloader class for an object detection dataset
    """
    def __init__(
        self, 
        dataset, 
        batch_size=32 
    ):  
        """
        Dataloader class constructor

        Parameters
        ----------
        dataset: object detection dataset from which to load the images
        batch_size: number of images to load at one time
        """
        self.data = dataset
        self.batch_size = batch_size
        if self.data.name == "COCO":
            self.image_format = "coco"
        elif self.data.name == "VOC":
            self.image_format = "pascal_voc"
        else:
            raise ValueError("Wrong or unsupported dataset. Available: 'COCO' or 'VOC'")
        self.transformation = Transform(image_dim=self.data.image_dim, image_format=self.image_format)

    def generate_batch(self, phase="train"):
        """
        Generate a train and a validation batch

        Parameters
        ----------
        phase: 'train', 'eval' or 'test'

        Return
        ------
        batch: list of dict, with fields:
            'image': image read from the dataset
            'labels': bboxes with label (Labeled_Box object)
        ids: ids of the images in the batch
        """
        if phase == "train":
            image_dir = os.path.join(self.data.train_dir, self.data.imgs_folder)
            generator_ids = self.data.train_ids
            generator_urls = self.data.train_urls
            generator_filenames = self.data.train_filenames
            if len(generator_urls) > 0:
                generator = list(zip(generator_ids, generator_urls, generator_filenames))
                random.shuffle(generator)
                generator_ids, generator_urls, generator_filenames = zip(*generator)
            else:
                generator = list(zip(generator_ids, generator_filenames))
                random.shuffle(generator)
                generator_ids, generator_filenames = zip(*generator)            
        elif phase == "eval":
            image_dir = os.path.join(self.data.val_dir, self.data.imgs_folder)
            generator_ids = self.data.val_ids
            generator_urls = self.data.val_urls
            generator_filenames = self.data.val_filenames
        elif phase == "test":
            image_dir = os.path.join(self.data.test_dir, self.data.imgs_folder)
            generator_ids = self.data.test_ids
            generator_urls = self.data.test_urls
            generator_filenames = self.data.test_filenames
        else:
            raise ValueError("Wrong value for arg 'phase'. Available 'train', 'eval' or 'test'")
        generator_len = len(generator_ids)
        batch_size = self.batch_size

        for index in range(0, generator_len, batch_size):
            indices = list(range(index, min(index + batch_size, generator_len)))
            ids = [generator_ids[i] for i in indices]
            pool = mp.Pool(processes=NUM_WORKERS)
            if self.data.image_source == "url":
                #print(" - Reading data from urls...")
                urls = [generator_urls[i] for i in indices]
                imgs = pool.map(self.read_url, urls)
            else:
                #print(" - Reading data from disk...")
                filenames = [os.path.join(image_dir, generator_filenames[i])
                             for i in indices]
                imgs = pool.map(self.read_img, filenames)       
            pool.close()
            pool.join()                    
            #print(" - Preprocessing images...")
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
        phase: 'train', 'eval' or 'test'
        normalize: if True, bboxes are normalized in range [0, 1]
                  and converted to list format (useful during train
                  for efficiency reasons)

        Return
        ------
        processed_images: list of preprocessed images
        processed_labels: list of preprocessed labels tensors, in the format:
            [[bbox1 | label1], [bbox2 | label2], ...], with bbox and label in the format: 
                [[x_min, y_min, x_max, y_max] | label]
        """
        processed_images, processed_labels = [], []
        for i in range(len(images)):
            img_index = self.get_img_index(ids[i], phase)
            image, bboxes, class_labels = None, None, None
            if len(images[i].shape) == 2 and self.data.n_channels == 3:
                images[i] = self.gray_to_rgb(images[i])
            if phase == "train":
                bboxes = self.data.train_bboxes[img_index]
                class_labels = self.data.train_labels[img_index]
            elif phase == "eval":
                bboxes = self.data.val_bboxes[img_index]
                class_labels = self.data.val_labels[img_index]
            else:
                bboxes = self.data.test_bboxes[img_index]
                class_labels = self.data.test_labels[img_index]
            img_transformed, bboxes_transformed, class_labels = \
                self.transformation(images[i], bboxes, class_labels, phase)
            image = (img_transformed / 127.5) - 1.
            bboxes = bboxes_transformed
            class_labels = [self.data.labels_dict[cl] for cl in class_labels]
            labeled_boxes = []
            if normalize:
                # normalize -> min_max coords, [0, 1] range, label added
                labeled_boxes = tf.stack([normalize_bbox(
                                              bboxes[i],
                                              self.data.image_dim[0], self.data.image_dim[1],
                                              class_labels[i],
                                              self.image_format
                                         ) for i in range(len(class_labels))], 0)
            else:
                for box, label in zip(bboxes_transformed, class_labels): 
                    labeled_boxes.append(Labeled_Box(bbox=Bounding_Box(*box), label=label))
            processed_images.append(image)
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