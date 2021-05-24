import tensorflow as tf
from params import *

def get_batch_keypoints(heatmaps):
    """
    heatmaps: (batch_size, height, width, n_keypoints)
    """
    assert len(heatmaps.shape) == 4

    width = tf.cast(tf.shape(heatmaps)[2], tf.int64)
    flattened = tf.reshape(heatmaps, shape=(tf.shape(heatmaps)[0], tf.shape(heatmaps)[3], -1))
    # mean = tf.reduce_mean(flattened, axis=-1)
    maxval = tf.reduce_max(flattened, axis=-1)
    maxind = tf.argmax(flattened, axis=-1)
    
    coord = tf.stack([(maxind % width)+1, (maxind // width) + 1], axis=-1)
    blank_preds = tf.ones_like(coord) * tf.constant(-1, dtype=tf.int64)

    return tf.where(tf.broadcast_to(tf.expand_dims(maxval > 0, axis=-1), tf.shape(coord)), coord, blank_preds)


def calc_dists(gt_kp, pred_kp):
    """
    gt_kp: (batch_size, n_keypoints, 2)
    pred_kp: (batch_size, n_keypoints, 2)
    
    """
    dists = tf.norm(tf.cast(gt_kp - pred_kp, tf.float64), ord='euclidean', axis=-1)
    return dists


def accuracy(y_true, y_pred, thr=pckh, n_keypoints=n_keypoints):
    """
    y_true: (batch_size, height, width, n_keypoints)
    y_pred: (batch_size, height, width, n_keypoints)

    Returns:
    total_acc/c: Total Percentage Accuracy across all keypoints

    (Optional) Alternatively, return
    accs: Percentage Accuracy of individual keypoints

    """
    gt_keypoints = get_batch_keypoints(y_true)
    pred_keypoints = get_batch_keypoints(y_pred)
    norm = tf.ones(tf.shape(pred_keypoints)[0], dtype=tf.float64) * tf.cast(tf.shape(y_true)[2], tf.float64) / 10.
    dists = calc_dists(gt_keypoints, pred_keypoints)
    
    norm_dists = dists / tf.broadcast_to(tf.expand_dims(norm, axis=-1), tf.shape(dists))
    
    missing_gt_mask = tf.logical_and(gt_keypoints[:,:,0] > 1, gt_keypoints[:,:,1] > 1)
    missing_preds = tf.ones_like(norm_dists)*-1
    valid_dists = tf.where(missing_gt_mask, norm_dists, missing_preds)
    
    
    accs = []
    total_acc = tf.constant(0, dtype=tf.float64)
    c = tf.constant(0, dtype=tf.float64)

    for i in range(n_keypoints):
        correct = tf.cast((valid_dists[:,i] <= thr), tf.float64)
        acc = tf.boolean_mask(correct, missing_gt_mask[:,i])
        accs.append(tf.reduce_mean(acc))
        if tf.reduce_mean(valid_dists[:,i]) > -1:
          total_acc += accs[i]
          c += 1

    accs = tf.stack(accs) 

    return total_acc / c