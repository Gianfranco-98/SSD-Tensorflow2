#!/usr/bin/env python


# Stock libraries
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, LayerNormalization

# My libraries
from configuration import *
from models import BaseNet, ExtraNet, DetectorNet


class SSD(Model):
    """
    Single-Shot multibox Detector (SSD) Class
    """
    def __init__(
        self, 
        num_classes, 
        input_shape, 
        default_boxes_num = DEFAULT_BOXES_NUM, 
    ):
        """
        SSD class constructor

        Parameters
        ----------
        num_classes: number of classes in the dataset, plus background class
        input_shape: shape of the images in input to the network
        default_boxes_num: number of prior default boxes
        """
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
            architecture=self.base_architecture,            
            input_shape=input_shape,
            name="SSD_Base"
        )
        self.l2_norm = LayerNormalization(
            epsilon=0.001,
            center=True,
            scale=True,
            beta_initializer='glorot_uniform',
            gamma_initializer='glorot_uniform'
        )
        self.extra_layers = ExtraNet(                       
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
        self.detector = DetectorNet(                        
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
        # ---------------------------------------------------------------------------------------------- #

    @property
    def output_shape(self):
        """
        Output shape of the SSD
        """
        return self.detector.output_shape

    def summary(self):
        """
        Output the complete structure of the network
        """
        self.base.summary()
        self.extra_layers.summary()
        self.detector.summary()

    def process_feature_maps(self, feature_maps):
        """
        Reshape the output of the ssd in order to have a prediction in the format:
            [BATCH_SIZE * TOT_PREDICTIONS * [NUM_CLASSES * l, x1, x2, x3, x4]]
                - x1, x2, x3, x4 are the coordinates (encoded) of the predicted bboxes
                - NUM_CLASSES * l are the predicted class scores
        """
        batch_size = feature_maps[0].shape[0]
        feature_maps_reshaped = []
        for feature in feature_maps:
            feature_maps_reshaped.append(
                tf.reshape(feature, shape=(batch_size, -1, self.num_classes+4)))
        prediction = tf.concat(feature_maps_reshaped, axis=1, name="SSD_prediction") 
        return prediction

    def call(self, inputs, training=False):
        """
        Call function of the SSD. Returns the feature maps
        """
        # BASE outputs
        x = self.base.indexable_call(inputs, end="block4_conv3")
        out4_3 = self.l2_norm(x)
        out7 = self.base.indexable_call(x, start="block4_pool")

        # EXTRA outputs
        out8_2 = self.extra_layers.indexable_call(out7, end="conv8_2")
        out9_2 = self.extra_layers.indexable_call(out8_2, start="conv9_1", end="conv9_2")
        out10_2 = self.extra_layers.indexable_call(out9_2, start="conv10_1", end="conv10_2")
        out11_2 = self.extra_layers.indexable_call(out10_2, start="conv11_1")

        # DETECTOR predictions  
        feature_maps = self.detector(inputs=[
            out4_3, out7, out8_2, out9_2, out10_2, out11_2])
        
        return feature_maps