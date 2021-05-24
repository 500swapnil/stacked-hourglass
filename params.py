import tensorflow as tf

# Evaluation Parameters
pckh = 3

# Model Parameters
AUTO = tf.data.experimental.AUTOTUNE
IMAGE_SIZE = (64,64)
n_keypoints = 10
n_blocks = 1
n_stacks = 2

# Train Parameters
anno_file = '../chair/chairdata.csv'
DATA_DIR = '../chair/'
base_lr = 2.5e-4
BATCH_SIZE = 8
val_split = 0.2
EPOCHS = 20
