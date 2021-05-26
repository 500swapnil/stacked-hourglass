import tensorflow as tf
from utils.misc import get_indices_from_slice
from params import *

# Augmentations
@tf.function
def random_lighting_noise(image):
    if tf.random.uniform([]) > 0.5:
        channels = tf.unstack(image, axis=-1)
        channels = tf.random.shuffle(channels)
        image = tf.stack([channels[0], channels[1], channels[2]], axis=-1)
    return image

@tf.function
def random_flip(image, keypoints, flip_prob=tf.constant(0.5), image_width=IMAGE_SIZE[1]):
    if tf.random.uniform([]) > flip_prob:
        mask = tf.equal(keypoints, -1)
        image = tf.image.flip_left_right(image)
        keypoints = tf.stack([image_width - keypoints[:,0], keypoints[:,1]], axis=1)
        # Remove -1 keypoints that are now image_width + 1
        keypoints = tf.where( mask, -1 * tf.ones_like(keypoints), keypoints )
    return (image, keypoints)

@tf.function
def expand(image, keypoints, expand_prob = 0.5):
    if tf.random.uniform([]) > expand_prob:
        return image, keypoints

    image_shape = tf.cast(tf.shape(image), tf.float32)
    ratio = tf.random.uniform([], 1, 2, dtype=tf.float32)
    left = tf.math.round(tf.random.uniform([], 0, image_shape[1]*ratio - image_shape[1]))
    top = tf.math.round(tf.random.uniform([], 0, image_shape[0]*ratio - image_shape[0]))
    new_height = tf.math.round(image_shape[0]*ratio)
    new_width = tf.math.round(image_shape[1]*ratio)
    expand_image = tf.zeros(( new_height, new_width , image_shape[2]), dtype=tf.float32)
    indices = get_indices_from_slice(int(top), int(left), int(image_shape[0]), int(image_shape[1]))
    expand_image = tf.tensor_scatter_nd_update(expand_image, indices, tf.reshape(image, [-1,3]))
    mask = tf.equal(keypoints, -1)
    image = expand_image
    x = (keypoints[:,0] + left) * image_shape[1] / new_width
    y = (keypoints[:,1] + top) * image_shape[0] / new_height
    
    keypoints = tf.round(tf.stack([x, y], axis=1))
    keypoints = tf.where( mask, -1 * tf.ones_like(keypoints), keypoints )
    image = tf.image.resize(image, IMAGE_SIZE)
    
    return image, keypoints