import os
from tqdm import tqdm
import time
import warnings

DATASET_NAME = 'VOC'
DATASET_KEY = '07+12'

# Create VOC2012 train dir
start = time.time()
os.mkdir('VOC2012')
os.mkdir('./VOC2012/JPEGImages')
os.mkdir('./VOC2012/Annotations')
if DATASET_KEY == '07++12':
    os.mkdir('./VOC2012/Test')
    os.mkdir('./VOC2012/Test/JPEGImages')
    os.mkdir('./VOC2012/Test/Annotations')
elif DATASET_KEY == '07+12':
    os.mkdir('VOC2007')
    os.mkdir('./VOC2007/Test')
    os.mkdir('./VOC2007/Test/JPEGImages')
    os.mkdir('./VOC2007/Test/Annotations')
else:
    raise ValueError("Datset key should be '07++12' or '07+12'")

# VOC trainval2012
os.chdir('./tmp/VOC2012/ImageSets/Main')
trainval_names = open("trainval.txt").read().split('\n')[:-1]
trainval_imgs_2012 = [name + '.jpg' for name in trainval_names]
os.chdir('../../JPEGImages')
print("\nAdding trainval2012 images...")
for filename in tqdm(os.listdir()):
    if filename in trainval_imgs_2012:
        os.rename(filename, '../../../VOC2012/JPEGImages/' + filename)
print("\nAdding trainval2012 annotations...")
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
print("\nAdding trianval2007 images...")
for filename in tqdm(os.listdir()):
    if filename in trainval_imgs_2007:
        os.rename(filename, '../../../VOC2012/JPEGImages/' + filename)
print("\nAdding trainval2007 annotations...")
os.chdir('../Annotations')
trainval_anns_2007 = []
for filename in tqdm(os.listdir()):
    if filename[:-4] in trainval_names:
        trainval_anns_2007.append(filename)
        os.rename(filename, '../../../VOC2012/Annotations/' + filename)
os.chdir('../../..')

# VOC test 2007
if DATASET_KEY == '07++12':
    imgs_2007_dst = '../../../VOC2012/JPEGImages/'
    anns_2007_dst = '../../../VOC2012/Annotations/'
elif DATASET_KEY == '07+12':
    imgs_2007_dst = '../../../VOC2007/Test/JPEGImages/'
    anns_2007_dst = '../../../VOC2007/Test/Annotations/'
else:
    # Insert alternative dataset configurations
    imgs_2007_dst, anns_2007_dst = None, None
os.chdir('./tmp/VOC2007_Test/ImageSets/Main')
test_names = open("test.txt").read().split('\n')[:-1]
test_imgs_2007 = [name + '.jpg' for name in test_names]
os.chdir('../../JPEGImages')
print("\nAdding test2007 images...")
for filename in tqdm(os.listdir()):
    if filename in test_imgs_2007:
        os.rename(filename, imgs_2007_dst + filename)
print("\nAdding test2007 annotations...")
os.chdir('../Annotations')
test_anns_2007 = []
for filename in tqdm(os.listdir()):
    if filename[:-4] in test_names:
        test_anns_2007.append(filename)
        os.rename(filename, anns_2007_dst + filename)
os.chdir('../../..')

# VOC test 2012
if DATASET_KEY == '07++12':
    os.chdir('./tmp/Test/ImageSets/Main')
    test_names = open("test.txt").read().split('\n')[:-1]
    test_imgs_2012 = [name + '.jpg' for name in test_names]
    os.chdir('../../JPEGImages')
    print("\nAdding test2012 images...")
    for filename in tqdm(os.listdir()):
        if filename in test_imgs_2012:
            os.rename(filename, '../../../VOC2012/Test/JPEGImages/' + filename)
    warnings.warn("Missing Test Annotations in VOC2012")
    print("\nAdding test2012 annotations...")   
    os.chdir('../Annotations')
    test_anns_2012 = []
    for filename in tqdm(os.listdir()):
        if filename[:-4] in test_names:
            test_anns_2012.append(filename)
            os.rename(filename, '../../../VOC2012/Test/Annotations/' + filename)
    os.chdir('../../..')

# Create train set
if DATASET_KEY == '07++12':
    print("\nCreating Train set = VOC2012 trainval + VOC2007 trainval + VOC2007 test...")
    train_imgs = trainval_imgs_2012 + trainval_imgs_2007 + test_imgs_2007
    train_anns = trainval_anns_2012 + trainval_anns_2007 + test_anns_2007
elif DATASET_KEY == '07+12':
    print("\nCreating Train set = VOC2012 trainval + VOC2007 trainval...")
    train_imgs = trainval_imgs_2012 + trainval_imgs_2007
    train_anns = trainval_anns_2012 + trainval_anns_2007
else:
    # Insert alternative dataset configurations
    train_imgs, train_anns = None, None
train_imgs.sort()
train_anns.sort()

# Create test set
if DATASET_KEY == '07++12':
    print("Creating Test set = VOC2012 test...")
    test_imgs = test_imgs_2012
    test_anns = test_anns_2012
elif DATASET_KEY == '07+12':
    print("Creating Test set = VOC2007 test...")
    test_imgs = test_imgs_2007
    test_anns = test_anns_2007
else:
    # Insert alternative dataset configurations
    test_imgs, test_anns = None, None
test_imgs.sort()
test_anns.sort()

print("Done in %f s" % (time.time() - start))