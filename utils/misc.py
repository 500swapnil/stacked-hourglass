import tensorflow as tf

@tf.function
def get_indices_from_slice(top, left, height, width):
    a = tf.reshape(tf.range(top, top + height),[-1,1])
    b = tf.range(left,left+width)
    A = tf.reshape(tf.tile(a,[1,width]),[-1])
    B = tf.tile(b,[height])
    indices = tf.stack([A,B], axis=1)
    return indices