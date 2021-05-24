import tensorflow as tf
from model.stacked_hourglass import StackedHourglass
import pandas as pd
from data import prepare_dataset
from params import *
from model.loss import hg_loss
from utils.evaluation import accuracy

df = pd.read_csv(anno_file, header=None)
filenames = df.pop(0)
filenames = DATA_DIR + filenames.astype(str)
DATASET_SIZE = len(filenames)
train_size = int((1 - val_split) * DATASET_SIZE)
val_size = int(val_split * DATASET_SIZE)
dataset = tf.data.Dataset.from_tensor_slices((filenames.values, df.values))

dataset = dataset.shuffle(1000)

train_dataset = dataset.take(train_size)
val_dataset = dataset.take(val_size)

train_dataset = prepare_dataset(train_dataset, BATCH_SIZE, train=True)
val_dataset = prepare_dataset(val_dataset, BATCH_SIZE)


hourglass = StackedHourglass(n_stacks=n_stacks, n_blocks=n_blocks, n_keypoints=n_keypoints)

hourglass.compile(
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_lr, centered=True),
    loss = hg_loss,
    metrics = [accuracy]
)

hourglass.build(input_shape=(BATCH_SIZE, *IMAGE_SIZE, 3))
hourglass.summary()

steps_per_epoch = train_size // BATCH_SIZE
validation_steps = val_size // BATCH_SIZE

history = hourglass.fit(train_dataset, 
                    validation_data=val_dataset,
                    steps_per_epoch=steps_per_epoch, 
                    validation_steps=validation_steps,
                    epochs=EPOCHS)



