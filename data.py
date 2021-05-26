import tensorflow as tf
from params import *
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from utils.heatmap import generate_heatmap
from utils.augmentations import *

@tf.function
def decode_img(img,  image_size=IMAGE_SIZE):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, image_size)

@tf.function
def prepare_input(file_path, keypoints, image_size=IMAGE_SIZE):
    img = tf.io.read_file(file_path)
    img = decode_img(img, image_size)
    img = preprocess_input(img, mode='torch')
    group_keypoints = tf.cast(tf.reshape(keypoints, shape=(n_keypoints, 2)), tf.float32)
    return img, group_keypoints

@tf.function
def prepare_heatmaps(image, keypoints, image_size=IMAGE_SIZE):
    heatmaps = []
    for i in range(n_keypoints):
        heatmap = generate_heatmap(keypoints[i], image_size)
        heatmaps.append(heatmap)
    return image, tf.stack(heatmaps, axis=-1)

@tf.function
def data_augment(image, keypoints):
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5) # Random Saturation
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.15) # Random brightness
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5) # Random Contrast
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_hue(image, max_delta=0.2) # Random Hue
    image = random_lighting_noise(image) # Random Lighting Noise
    image, keypoints = expand(image, keypoints) # Random Scaling
    image, keypoints = random_flip(image, keypoints) # Random Flip

    return (image, keypoints)

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
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.map(prepare_heatmaps)
    dataset = dataset.padded_batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset