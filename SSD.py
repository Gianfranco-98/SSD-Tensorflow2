#!/usr/bin/env python3

from models import BaseNet, ExtraNet, DetectorNet
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Input


BASE_NAME = "VGG16"
DEFAULT_BOXES_NUM = [4, 6, 6, 6, 4, 4]
INPUT_SHAPE = (300, 300, 3)


class SSD(Model):

    def __init__(
        self, 
        num_classes, 
        input_shape, 
        default_boxes_num = DEFAULT_BOXES_NUM, 
    ):
        super(SSD, self).__init__()
        self.base_architecture = BASE_NAME
        self.num_classes = num_classes
        self.default_boxes_num = DEFAULT_BOXES_NUM
        self.filters_num = \
            [(self.num_classes + 4)*def_num for def_num in self.default_boxes_num]
        
        # ---------------------------------------------------------------------------------------------- #
        # ---------------------------------------| SSD Structure |-------------------------------------- # 
        # ---------------------------------------------------------------------------------------------- #
        self.base = BaseNet(
            architecture=self.base_architecture,            #TODO: fine-tuning
            input_shape=input_shape,
            name="SSD_Base"
        )
        self.extra_layers = ExtraNet(                       #TODO: Xavier initialization
            layers = [
                Input(shape=self.base[-1].output_shape[1:], name="input_extra"),
                ## Conv 8 
                Conv2D(256, kernel_size=1, strides=1, padding="valid", activation='relu', name="conv8_1"),
                Conv2D(512, kernel_size=3, strides=2, padding="same", activation='relu', name="conv8_2"),
                ## Conv 9 
                Conv2D(128, kernel_size=1, strides=1, padding="valid", activation='relu', name="conv9_1"),
                Conv2D(256, kernel_size=3, strides=2, padding="same", activation='relu', name="conv9_2"),
                ## Conv 10 
                Conv2D(128, kernel_size=1, strides=1, padding="valid", activation='relu', name="conv10_1"),
                Conv2D(256, kernel_size=3, strides=1, padding="valid", activation='relu', name="conv10_2"),
                ## Conv 11
                Conv2D(128, kernel_size=1, strides=1, padding="valid", activation='relu', name="conv11_1"),
                Conv2D(256, kernel_size=3, strides=1, padding="valid", activation='relu', name="conv11_2")
            ],
            name="SSD_Extra_Layers"
        )
        self.detector = DetectorNet(                        #TODO: Xavier initialization
            input_layers = [
                Input(shape=self.base["block4_conv3"].output_shape[1:], name="input_predict4_3"),
                Input(shape=self.base["head_conv7"].output_shape[1:], name="input_predict7"),
                Input(shape=self.extra_layers["conv8_2"].output_shape[1:], name="input_conv8_2"),
                Input(shape=self.extra_layers["conv9_2"].output_shape[1:], name="input_conv9_2"),
                Input(shape=self.extra_layers["conv10_2"].output_shape[1:], name="input_conv10_2"),
                Input(shape=self.extra_layers["conv11_2"].output_shape[1:], name="input_conv11_2")
            ],
            predictors = [
                Conv2D(self.filters_num[0], kernel_size=3, strides=1, padding="same", name="predict4_3"),
                Conv2D(self.filters_num[1], kernel_size=3, strides=1, padding="same", name="predict7"),
                Conv2D(self.filters_num[2], kernel_size=3, strides=1, padding="same", name="predict8_2"),
                Conv2D(self.filters_num[3], kernel_size=3, strides=1, padding="same", name="predict9_2"),
                Conv2D(self.filters_num[4], kernel_size=3, strides=1, padding="same", name="predict10_2"),
                Conv2D(self.filters_num[5], kernel_size=3, strides=1, padding="same", name="predict11_2")
            ],
            name="SSD_Detector"
        )

        #TODO: non-maximum suppression

        # ---------------------------------------------------------------------------------------------- #
    

    @property
    def layers(self):
        return (
            self.base.layers +
            self.extra_layers.layers +
            self.detector.layers
        )

    def summary(self):
        self.base.summary()
        self.extra_layers.summary()
        self.detector.summary()

    def process_feature_maps(self, feature_maps):
        batch_size = feature_maps[0].shape[0]
        feature_maps_reshaped = []
        for feature in feature_maps:
            feature_maps_reshaped.append(
                tf.reshape(feature, shape=(batch_size, -1, self.num_classes+4)))
        prediction = tf.concat(feature_maps_reshaped, axis=1, name="SSD_prediction") 
        return prediction

    def call(self, inputs, training=False):

        # BASE outputs
        x = self.base[0:"block4_conv3"](inputs)
        out4_3 = tf.math.l2_normalize(x=x, axis=-1)
        out7 = self.base["block4_pool":](x)

        # EXTRA outputs
        out8_2 = self.extra_layers[0:"conv8_2"](out7)
        out9_2 = self.extra_layers["conv9_1":"conv9_2"](out8_2)
        out10_2 = self.extra_layers["conv10_1":"conv10_2"](out9_2)
        out11_2 = self.extra_layers["conv11_1":](out10_2)

        # DETECTOR predictions  
        feature_maps = self.detector(inputs=[
            out4_3, out7, out8_2, out9_2, out10_2, out11_2])
        
        return feature_maps
