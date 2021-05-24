import tensorflow as tf

def hg_loss(y_true, y_pred):
    """
    y_true: (batch_size, height, width, n_keypoints)
    y_pred: (batch_size, n_stacks, height, width, n_keypoints)
    """
    mse = tf.keras.losses.MeanSquaredError()
    loss = 0.
    for stack_pred in y_pred:
        loss += mse(y_true, stack_pred)
    
    return loss
