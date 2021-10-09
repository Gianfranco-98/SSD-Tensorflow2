#!/usr/bin/env python


# Stock libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from shutil import copyfile

# My libraries
from configuration import *


def learn(ssd, ssd_optimizer, ssd_loss, images, matched_boxes, labels):
    """
    Take a learning step by predict, compute loss and optimize

    Parameters
    ----------
    ssd_optimizer: optimizer of the SSD network
    ssd_loss: ssd loss function / callable class
    images: images to pass into the network
    matched_boxes: default boxes matched with gt boxes and encoded
    labels: ground truth labels
    """
    with tf.GradientTape() as ssd_tape:
        # Prediction
        feature_maps = ssd(images)
        ssd_prediction = ssd.process_feature_maps(feature_maps)
        # Loss
        multibox_loss, localization_loss, confidence_loss = \
            ssd_loss(
                gt_bboxes=matched_boxes,
                gt_classes=labels,
                prediction=ssd_prediction,
            )
        # Weight Decay
        l2_loss = [tf.nn.l2_loss(t) for t in ssd.trainable_variables]
        l2_loss = WEIGHT_DECAY * tf.math.reduce_sum(l2_loss)
        multibox_loss += l2_loss
    # Optimization
    ssd_gradients = ssd_tape.gradient(multibox_loss, ssd.trainable_variables)
    ssd_optimizer.apply_gradients(zip(ssd_gradients, ssd.trainable_variables))
    return multibox_loss, localization_loss, confidence_loss


def save_train_data(checkpoint_handler, checkpoint_filepath, checkpoint_dir, iterations, mb_losses, loc_losses, conf_losses):
    """
    Save network weights and training informations into np arrays

    Parameters
    ----------
    checkpoint_handler: tensorflow checkpoint handler to save the network weights
    checkpoint_filepath: filepath to save the network weights
    iterations: array with actual iterations for each iteration
    mb_losses: array of mulibox losses 
    loc_losses: array of localization losses
    conf_losses: array of confidence losses
    """
    # Save SSD weights
    save_path = checkpoint_handler.save(checkpoint_filepath + str(iterations[-1]) + '_iter')

    # Convert losses data to Numpy Array
    mb = np.array(mb_losses)
    loc = np.array(loc_losses)
    conf = np.array(conf_losses)

    # Convert all to binary file
    np.save('./iterations', np.array(iterations))
    np.save('./mb_losses', mb)
    np.save('./loc_losses', loc)
    np.save('./conf_losses', conf)

    # Move files to a drive folder
    copyfile('./iterations.npy', checkpoint_dir + '/train_data/iterations.npy')
    copyfile('./mb_losses.npy', checkpoint_dir + '/train_data/mb_losses.npy')
    copyfile('./loc_losses.npy', checkpoint_dir + '/train_data/loc_losses.npy')
    copyfile('./conf_losses.npy', checkpoint_dir + '/train_data/conf_losses.npy')


def load_train_data(net, checkpoint_dir):
    """
    Load training data like network weights and training arrays

    Parameters
    ----------
    checkpoint_dir: directory where are stored the training data
    
    Return
    ------
    net: new network with loaded weights
    iterations: array with actual iterations for each iteration
    mb_losses: array of mulibox losses 
    loc_losses: array of localization losses
    conf_losses: array of confidence losses
    """
    # Load SSD weights
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    net.load_weights(latest)

    # Copy iterations and losses data into actual scope
    copyfile(checkpoint_dir + '/train_data/iterations.npy', './iterations.npy')
    copyfile(checkpoint_dir + '/train_data/mb_losses.npy', './mb_losses.npy')
    copyfile(checkpoint_dir + '/train_data/loc_losses.npy', './loc_losses.npy')
    copyfile(checkpoint_dir + '/train_data/conf_losses.npy', './conf_losses.npy')

    # Load iterations and losses data
    iterations = np.load('./iterations.npy').tolist()
    mb_losses = np.load('./mb_losses.npy').tolist()
    loc_losses = np.load('./loc_losses.npy').tolist()
    conf_losses = np.load('./conf_losses.npy').tolist()

    return net, iterations, mb_losses, loc_losses, conf_losses


def plot_train_data(iterations, mb_losses, loc_losses, conf_losses):
    """
    Plot a graph with the training trend

    Parameters
    ----------
    iterations: array with actual iterations for each iteration
    mb_losses: array of mulibox losses 
    loc_losses: array of localization losses
    conf_losses: array of confidence losses
    """
    # Plot the three losses
    fig = plt.figure()

    plt.title('Training Process')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    l1, = plt.plot(iterations, mb_losses, c='green')
    l2, = plt.plot(iterations, loc_losses, c='red')
    l3, = plt.plot(iterations, conf_losses, c='blue')

    plt.legend(handles=[l1, l2, l3], labels=['Multibox Loss', 'Localization Loss', 'Confidence Loss'], loc='best')
    plt.show()

    # Plot specifically the Localization Loss
    fig = plt.figure()

    plt.title('Localization Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    l1, = plt.plot(iterations, loc_losses, c='red')

    plt.legend(handles=[l1], labels=['Localization Loss'], loc='best')
    plt.show()
