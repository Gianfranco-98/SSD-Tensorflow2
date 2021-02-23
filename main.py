#!/usr/bin/env python3

# ___________________________________________________ Libraries ___________________________________________________ #


import cv2
from cv2 import imread
from skimage import io
import matplotlib.pyplot as plt
from image_detection import Dataloader, Image_Element
from COCO_utils import *

# ___________________________________________________ Constants ___________________________________________________ #

# ___________________________________________________ Functions ___________________________________________________ #

# ______________________________________________________ Main ______________________________________________________ #


if __name__ == '__main__':
    
    dataloader = Dataloader(dataset="COCO")
    generator = dataloader.generate_batch()
    train_coco = dataloader.train_coco
    val_coco = dataloader.val_coco
    content = get_image_content(val_coco, 448263)
    print(content)
    content = [Image_Element(**elem) for elem in content]
    print(content)
    pause = input("Press Enter to continue...")
    """
    print("IMAGE EXAMPLES\n")
    for i in range(10):
        train_batch, val_batch = next(generator)
        plt.imshow(train_batch[0])
        plt.show()
        pause = input("Press Enter to continue...")
        plt.imshow(train_batch[-1])
        plt.show()
        pause = input("Press Enter to continue...")
        plt.imshow(val_batch[0])
        plt.show()
        pause = input("Press Enter to continue...")
        plt.imshow(val_batch[-1])
        plt.show()
        pause = input("Press Enter to continue...")
    """