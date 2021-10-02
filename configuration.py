# ______________________________________ Generic Parameters ______________________________________ #


# Files parameters
CHECKPOINT_DIR = './Checkpoints/VOC'
CHECKPOINT_FILEPATH = CHECKPOINT_DIR + '/checkpoint'

# Network parameters
BASE_WEIGHTS = 'imagenet'
BASE_NAME = "VGG16"
ASPECT_RATIOS = [[1., 2., 1/2],
                [1., 2., 1/2, 3., 1/3],
                [1., 2., 1/2, 3., 1/3],
                [1., 2., 1/2, 3., 1/3],
                [1., 2., 1/2],
                [1., 2., 1/2]]
DEFAULT_BOXES_NUM = [4, 6, 6, 6, 4, 4]
INPUT_SHAPE = (300, 300, 3)
IMAGE_DIM = (300, 300)
N_CHANNELS = 3

# Learning parameters
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
ALPHA = 1
REGRESSION_TYPE = 'smooth_l1'

# Train 
CHECKPOINT_PERIOD = 250
PLOT_PERIOD = 500
BATCH_SIZE = 32
NUM_WORKERS = 8
LOAD_MODEL = False
TENSORBOARD_LOGS = False

# Test
TEST_SIZE = 1
TEST_ITERATIONS = 10

# Data augmentation
IOU_THRESHOLDES = [0., 0.1, 0.3, 0.5, 0.7, 0.9]
PATCHFIND_ATTEMPTS = 50

# Non-Maximum Suppression
CONFIDENCE_THRESHOLD = 0.01
JACCARD_THRESHOLD = 0.45
MAX_NMS_BOXES = 200
TOP_K_BOXES = 10

# ____________________________________ Dataset Configuration ____________________________________ #


DATASET_NAME = "VOC"
DATASET_YEAR = "2012"
TESTSET_YEAR = "2007"
DATASET_KEY = "07+12"
DATA_PATH = './data/'
TRAINVAL_PATH = DATA_PATH + DATASET_NAME + DATASET_YEAR
TEST_PATH = DATA_PATH + DATASET_NAME + TESTSET_YEAR + '/Test'
TRAIN_ANN_PATH = None
VAL_ANN_PATH = None
ANN_PATH = None

if DATASET_NAME == "COCO":
    TRAIN_ANN_PATH = TRAINVAL_PATH + '/annotations/instances_train' + DATASET_YEAR + '.json'
    VAL_ANN_PATH = TRAINVAL_PATH + '/annotations/instances_val' + DATASET_YEAR + '.json'
    TEST_ANN_PATH = TEST_PATH + '/Annotations'
    SCALES = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
    LR_VALUES = [1e-3, 1e-4, 1e-5, 1e-5]
    BOUNDARIES = [160000, 200000, 240000]
elif DATASET_NAME == "VOC":
    IMGS_FOLDER = 'JPEGImages'
    ANNS_FOLDER = 'Annotations'
    TRAIN_ANN_PATH = TRAINVAL_PATH + '/' + ANNS_FOLDER
    VAL_ANN_PATH = TEST_PATH + '/' + ANNS_FOLDER
    TEST_ANN_PATH = VAL_ANN_PATH
    SCALES = [0.10, 0.20, 0.37, 0.54, 0.71, 0.88, 1.05]
    LR_VALUES = [1e-3, 1e-4, 1e-4]
    BOUNDARIES = [60000, 80000]
ITERATIONS = max(BOUNDARIES)