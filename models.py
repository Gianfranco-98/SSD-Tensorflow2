#!/usr/bin/env python

# Stock libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import MaxPool2D, Conv2D

# My libraries
from base_models.vgg16_new import VGG16


# __________________________________ Support Classes __________________________________ #


class Network_Indexer:
    """
    Class useful for network indexing
    """
    def __init__(self):
        pass

    def get_item(self, layers, key):
        item = None
        if isinstance(key, int) or isinstance(key, str):
            item = layers[self.get_layer_index(layers, key)]
        elif isinstance(key, slice):
            start, stop = key.start, key.stop
            if isinstance(start, str):
                start = self.get_layer_index(layers, start)
            if isinstance(stop, str):
                stop = self.get_layer_index(layers, stop) + 1
            item = layers[start:stop]
            if len(item) > 1:
                item = Sequential(item)
        else:
            raise IndexError("Index must be int, str or slice")
        return item

    def get_layer_index(self, layers, key):
        """
        Return the layer index based on 'key' (int or name of the layer)
        """
        index = None
        if isinstance(key, int):
            if key >= len(layers):
                raise IndexError("Layers out of index. Must have 0 <= key < len(self.layers)")
            if key < 0:
                index = len(layers) + key
            index = key
        elif isinstance(key, str):
            for l, layer in zip(range(len(layers)), layers):
                if layer.name == key:
                    index = l
            if index is None:
                raise ValueError("No layer named '%s'" % key)
        else:
            raise TypeError("'key' must be an int or str")
        return index


class Powered_Sequential(Sequential):
    """
    Sequential class able to handle indexing and more features
    """
    def __init__(
        self,
        layers,
        name="Powered_Sequential",
    ):
        super(Powered_Sequential, self).__init__(layers=layers, name=name)
        self.indexer = Network_Indexer()

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, key):
        return self.indexer.get_item(self.layers, key)

    def indexable_call(self, inputs, training=False, start=None, end=None):
        """
        Normal sequential call, but with the possibility to choose first and last layer
        """
        outputs = None
        if start is None:
            start = 0
        if end is None:
            end = len(self.layers)
        if start is not None or end is not None:
            for l, layer in enumerate(self.layers):
                if l == start or layer.name == start:
                    start = l
                if l+1 == end or layer.name == end:
                    end = l+1
        for l in range(start, end):
            outputs = self.layers[l](inputs)
            inputs = outputs

        return outputs

    def call(self, inputs, training=False, start=None, end=None):
      pass

class Powered_Model(Model):
    """
    Model class able to handle indexing and more features
    """
    def __init__(
        self, 
        inputs, 
        outputs,
        name="Powered_Model", 
    ): 
        super(Powered_Model, self).__init__(inputs=inputs, outputs=outputs, name=name)
        self.indexer = Network_Indexer()

    def __len__(self):
        return len(self.layers) 

    def __getitem__(self, key):
        return self.indexer.get_item(self.layers, key)

    def indexable_call(self, inputs, training=False, start=None, end=None):
        """
        Normal model call, but with the possibility to choose first and last layer
        """
        outputs = None
        if start is None:
            start = 0
        if end is None:
            end = len(self.layers)
        if start is not None or end is not None:
            for l, layer in enumerate(self.layers):
                if l == start or layer.name == start:
                    start = l
                if l+1 == end or layer.name == end:
                    end = l+1
        for l in range(start, end):
            outputs = self.layers[l](inputs)
            inputs = outputs

        return outputs

    def call(self, inputs, training=False, start=None, end=None):
      pass


# __________________________________ Network Classes __________________________________ #


class BaseNet(Powered_Sequential):
    """
    Base Architecture Model, composed by a pre-trained net and few added head layers
    """
    def __init__(self, architecture, input_shape, name="BaseNet"):
        """
        Base Network class constructor

        Parameters
        ----------
        architecture: the name of the architecture for the base network
        input_shape: the shape of the images in input to the complete SSD network
        name: the name of this sub-network
        """
        if architecture in ["VGG16", "VGG-16", "VGG_16"]:

            # 1.1. Copy VGG16 pre-trained network
            arch = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
            base_layers = arch.layers[0:-1]

            # 1.2. Copy VGG16 head layers
            dummy_arch = VGG16(include_top=True, weights='imagenet') 
            head_layers = dummy_arch.layers[-3:]

            # 1.3 Copy old fc6, fc7 weights into new head layers
            fc6_weights, fc6_biases = head_layers[0].get_weights()
            fc7_weights, fc7_biases = head_layers[1].get_weights()

            # 2. Add new head
            base_layers.append(MaxPool2D(pool_size=3, strides=1, padding="same", name=arch.layers[-1].name))
            base_layers.append(Conv2D(1024, kernel_size=3, padding="same", dilation_rate=6, activation='relu', name="head_conv6",
                                      weights=[np.random.choice(np.reshape(fc6_weights, (-1,)), (3, 3, 512, 1024)),
                                               np.random.choice(fc6_biases, (1024,))]))
            base_layers.append(Conv2D(1024, kernel_size=1, padding="same", activation='relu', name="head_conv7",
                                      weights=[np.random.choice(np.reshape(fc7_weights, (-1,)), (1, 1, 1024, 1024)),
                                               np.random.choice(fc7_biases, (1024,))]))

            # 3. Instantiate model
            super(BaseNet, self).__init__(
                layers=base_layers,
                name=name
            )

        else:
            raise TypeError("Wrong name for the base architecture")


class ExtraNet(Powered_Sequential):
    """
    Extra Layers Model
    """
    def __init__(self, layers, name="ExtraNet"):
        """
        Extra Layers model constructor

        Parameters
        ----------
        layers: extra layers of the SSD network
        name: name of this sub-network
        """
        super(ExtraNet, self).__init__(
            layers=layers, 
            name=name
        )


class DetectorNet(Powered_Model):
    """
    Detector Model, composed by a list of input and corresponding predictors
    """
    def __init__(self, input_layers, predictors, name="DetectorNet"):
        """
        Detector model constructor

        Parameters
        ----------
        input_layers: list of input layers which are the input to the various detectors
        predictors: list of detector layers
        """
        outputs = [predictors[i](input_layers[i]) for i in range(len(predictors))]
        super(DetectorNet, self).__init__(
            inputs=input_layers, 
            outputs=outputs,
            name=name,
        )
        self.input_layers = input_layers
        self.predictors = predictors

    def __len__(self):
        return len(self.layers)

    @property
    def output_shape(self):
        """
        Output shapes of predictors
        """
        return [pred.output_shape for pred in self.predictors]

    def call(self, inputs, training=False):
        outputs = []
        for i in range(len(inputs)):
            in_predict = self.layers[i](inputs[i])
            out_layer = i + len(self.input_layers)
            outputs.append(super(DetectorNet, self).indexable_call(
                in_predict,
                start=out_layer,
                end=out_layer+1
            ))
        return outputs