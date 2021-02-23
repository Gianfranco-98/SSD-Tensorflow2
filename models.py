#!/usr/bin/env python3

import warnings
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from base_models.vgg16_new import VGG16
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import InputLayer, MaxPool2D, Conv2D, Layer, Input
from tensorflow.keras.models import clone_model
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops.numpy_ops import np_arrays


BASE_WEIGHTS = 'imagenet'


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

    def indexable_call(self, layers, inputs, start_layer=None, last_layer=None):
        """
        Indexable call for neural networks

        Parameters
        ----------
            inputs: tensor (np_array) or list of tensors (np_arrays)
            layers: layers of the model
            start_layer: layer from which to start forwarding the input
            last_layer: layer where to stop the forwarding

        Returns
        -------
            result of network forwarding operation
        """
        outputs = None
        start, last = 0, len(layers)-1
        if start_layer is not None:
            start = self.get_layer_index(layers, start_layer)
        if last_layer is not None:
            last = self.get_layer_index(layers, last_layer)
        for l in range(last+1):
            if l < start:
                continue
            outputs = layers[l](inputs)
            if l == last:
                break
            inputs = outputs
        return outputs


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

    def call(self, inputs, training=False, start_layer=None, last_layer=None):
        return self.indexer.indexable_call(
            self.layers, inputs, start_layer, last_layer)


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

    def call(self, inputs, training=False, start_layer=None, last_layer=None):
        return self.indexer.indexable_call(
            self.layers, inputs, start_layer, last_layer)


# __________________________________ Network Classes __________________________________ #


class BaseNet(Powered_Sequential):
    """
    Base Architecture Model, composed by a pre-trained net and few added head layers
    """
    def __init__(self, architecture, input_shape, name="BaseNet"):

        if architecture == "VGG16" or architecture == "VGG-16":    

            # 1. Copy VGG pre-trained network
            arch = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
            layers = arch.layers[0:-1]

            # 2. Modify layer "block3_pool in order to respect original paper"
            #TODO

            # 3. Add new head
            layers.append(MaxPool2D(pool_size=3, strides=1, padding="same", name=arch.layers[-1].name))
            layers.append(Conv2D(1024, kernel_size=3, padding="same", dilation_rate=6, activation='relu', name="head_conv6"))
            layers.append(Conv2D(1024, kernel_size=1, padding="same", activation='relu', name="head_conv7"))
            super(BaseNet, self).__init__(
                layers=layers,
                name=name
            )
            self.summary()

        else:
            raise TypeError("Wrong name for the base architecture")


class ExtraNet(Powered_Sequential):
    """
    Extra Layers Model
    """
    def __init__(self, layers, name="ExtraNet"):
        super(ExtraNet, self).__init__(
            layers=layers, 
            name=name
        )


class DetectorNet(Powered_Model):
    """
    Detector Model, composed by a list of input and corresponding predictors
    """
    def __init__(self, input_layers, predictors, name="DetectorNet"):
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

    def call(self, inputs, training=False):
        outputs = []
        for i in range(len(inputs)):
            in_predict = self.layers[i](inputs[i])
            out_layer = i + len(self.input_layers)
            outputs.append(super(DetectorNet, self).call(
                in_predict,
                start_layer=out_layer,
                last_layer=out_layer
            ))
        return outputs


"""
# TEST MAIN #
if __name__ == "__main__":

    base = BaseNet("VGG16")
    input1 = tf.expand_dims(tf.ones(shape=INPUT_SHAPE), 0)
    base.summary()
    print(base(input1, last_layer="head_conv7"))
"""
