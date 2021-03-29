#!/usr/bin/env python

import os
from tqdm import tqdm
import time

if __name__ == "__main__":

    # Create VOC2012 train dir
    start = time.time()
    os.mkdir('VOC2012')
    os.chdir('VOC2012')
    os.mkdir('JPEGImages')
    os.mkdir('Annotations')
    os.chdir('..')
    
    # VOC trainval2012
    os.chdir('./tmp/VOC2012/ImageSets/Main')
    trainval_names = open("trainval.txt").read().split('\n')[:-1]
    trainval_imgs_2012 = [name + '.jpg' for name in trainval_names]
    os.chdir('../../JPEGImages')
    print("Adding trainval2012 images...")
    for filename in tqdm(os.listdir()):
        if filename in trainval_imgs_2012:
            os.rename(filename, '../../../VOC2012/JPEGImages/' + filename)
    print("Adding trainval2012 annotations...")
    os.chdir('../Annotations')
    trainval_anns_2012 = []
    for filename in tqdm(os.listdir()):
        if filename[:-4] in trainval_names:
            trainval_anns_2012.append(filename)
            os.rename(filename, '../../../VOC2012/Annotations/' + filename)
    os.chdir('../../..')

    # VOC trainval2007
    os.chdir('./tmp/VOC2007/ImageSets/Main')
    trainval_names = open("trainval.txt").read().split('\n')[:-1]
    trainval_imgs_2007 = [name + '.jpg' for name in trainval_names]
    os.chdir('../../JPEGImages')
    print("Adding trianval2007 images...")
    for filename in tqdm(os.listdir()):
        if filename in trainval_imgs_2007:
            os.rename(filename, '../../../VOC2012/JPEGImages/' + filename)
    print("Adding trainval2007 annotations...")
    os.chdir('../Annotations')
    trainval_anns_2007 = []
    for filename in tqdm(os.listdir()):
        if filename[:-4] in trainval_names:
            trainval_anns_2007.append(filename)
            os.rename(filename, '../../../VOC2012/Annotations/' + filename)
    os.chdir('../../..')

    # VOC test 2007
    os.chdir('./tmp/VOC2007_Test/ImageSets/Main')
    test_names = open("test.txt").read().split('\n')[:-1]
    test_imgs_2007 = [name + '.jpg' for name in test_names]
    os.chdir('../../JPEGImages')
    print("Adding test2007 images...")
    for filename in tqdm(os.listdir()):
        if filename in test_imgs_2007:
            os.rename(filename, '../../../VOC2012/JPEGImages/' + filename)
    print("Adding test2007 annotations...")
    os.chdir('../Annotations')
    test_anns_2007 = []
    for filename in tqdm(os.listdir()):
        if filename[:-4] in test_names:
            test_anns_2007.append(filename)
            os.rename(filename, '../../../VOC2012/Annotations/' + filename)
    os.chdir('../../..')

    # Mix them up
    print("Mix all...")
    train_imgs = trainval_imgs_2012 + trainval_imgs_2007 + test_imgs_2007
    train_anns = trainval_anns_2012 + trainval_anns_2007 + test_anns_2007
    print("Done in %f s" % (time.time() - start))
