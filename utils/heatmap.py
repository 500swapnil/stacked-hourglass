import tensorflow as tf
from utils.misc import get_indices_from_slice
from params import *

@tf.function
def gaussian(size=7, sigma=1):
    x = tf.range(0,size,1,float)
    y = tf.expand_dims(x, 1)
    x0 = y0 = size // 2
    g = tf.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return g

@tf.function   
def generate_heatmap(center, img_size=IMAGE_SIZE, sigma=1):
    blank_heatmap = tf.zeros((img_size[0], img_size[1]))
    
    size = 6 * sigma + 1
    gauss = gaussian(size, sigma)

    top = tf.maximum(center[1]-size//2, 0)
    left = tf.maximum(center[0]-size//2, 0)
    bottom = tf.minimum(center[1]+size//2+1, img_size[0])
    right = tf.minimum(center[0]+size//2+1, img_size[1])

    gauss_top = tf.cast(tf.maximum(-(center[1]-size//2), 0), dtype=tf.int32)
    gauss_left = tf.cast(tf.maximum(-(center[0]-size//2), 0), dtype=tf.int32)
    gauss_bottom = tf.cast(tf.minimum(size - (center[1]+size//2+1 - img_size[0]), size), dtype=tf.int32)
    gauss_right = tf.cast(tf.minimum(size - (center[0]+size//2+1 - img_size[1]), size), dtype=tf.int32)

    indices = get_indices_from_slice(int(top), int(left), int(bottom - top), int(right - left))
    heatmap = tf.tensor_scatter_nd_update(blank_heatmap, indices, tf.reshape(gauss[gauss_top:gauss_bottom, gauss_left:gauss_right],[-1]))
    center_check = tf.logical_and(tf.greater(center[0],0), tf.greater(center[1],0))
    heatmap = tf.cond(center_check, lambda: heatmap, lambda: blank_heatmap)
    return heatmap
