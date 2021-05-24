import tensorflow as tf
from params import *
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from utils.heatmap import generate_heatmap

@tf.function
def decode_img(img,  image_size=[64, 64]):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, image_size)

@tf.function
def prepare_input(file_path, keypoints, image_size=[64, 64]):
    img = tf.io.read_file(file_path)
    img = decode_img(img, image_size)
    img = preprocess_input(img, mode='torch')
    labels = prepare_labels(keypoints)
    return img, labels

@tf.function
def prepare_labels(keypoints, image_size=[64,64]):
    group_keypoints = tf.stack(tf.split(keypoints, n_keypoints))
    heatmaps = []
    for i in range(n_keypoints):
        heatmap = generate_heatmap(group_keypoints[i], image_size)
        heatmaps.append(heatmap)
    return tf.stack(heatmaps, axis=-1)

@tf.function
def prepare_dataset(dataset, batch_size, train=False):
    # dataset = dataset.cache() # This dataset fits in RAM
    dataset = dataset.map(prepare_input, num_parallel_calls=AUTO)
    
    if train:
        # Best practices for Keras:
        # Training dataset: repeat then batch
        # Evaluation dataset: do not repeat
        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()
    dataset = dataset.padded_batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset