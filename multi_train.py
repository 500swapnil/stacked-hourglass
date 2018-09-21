import tensorflow as tf
import os
import cv2
import pandas as pd
import numpy as np
import time
import multi_hourglass as model
from datagenerator import DataGenerator
from progressBar import progBar

num_gpus = 4
datagen = DataGenerator(img_dir='chair3_final', nKeypoints=10, data_file='chair.txt')
model.save_dir = './saved_model/'
model.nFeats = 256
model.nStacks = 8
momentum = 0

def peak(heatmap):
    argmax = np.argmax(heatmap)
    return argmax % heatmap.shape[0], argmax // heatmap.shape[0]

def error(a, b, threshold=4):
    a_x, a_y = peak(a)
    b_x, b_y = peak(b)
    if b_x == 0 and b_y == 0:
        return -1
    return np.sqrt(np.square(a_x - b_x) + np.square(a_y - b_y)) <= threshold

def accur(prediction, gt_map, batch_size):
    total_correct = 0
    count_existing = 0
    for i in range(batch_size):
        hm_correct = error(prediction[i], gt_map[i])
        if hm_correct != -1:
            count_existing += 1
            total_correct += hm_correct

    if count_existing > 0:
        return total_correct/count_existing
    else:
        return 0.5

def total_accuracy(output, gt_map, nKeypoints, nStacks, batch_size):
    accur_array = []
    for i in range(nKeypoints):
        accur_array.append(accur(output[:, nStacks-1, :, :, i], gt_map[:, nStacks-1, :, :, i], batch_size))
    return accur_array

def tower_loss(scope, images, gt_map, weights):
    output = model.inference(images)
    _ = model.loss(output, gt_map, weights)
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')
    return total_loss

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        grads = []
        for g,_ in grad_and_vars:
            
            expanded_g = tf.expand_dims(g, 0)

            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train(nEpochs=10, epoch_size=None, is_restore=False, batch_size=8,
        learning_rate=2.5e-4, lr_decay=1, decay_step=2000, opt='rms'):
    
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)

        # if epoch_size is None:
        #     epoch_size = len(datagen.images) // batch_size
        # dummy_inp, dummy_gt, dummy_wt = datagen.generate_batch(batch_size=batch_size, nStacks=model.nStacks)
        # dummy_out = model.inference(dummy_inp)
        # saver = tf.train.Saver()
        
        total_start = time.time()
        
        lr = tf.train.exponential_decay(
            learning_rate, global_step, decay_step, lr_decay, staircase=True)

        if opt == 'rms':
            optim = tf.train.RMSPropOptimizer(learning_rate=lr)
        else:
            optim = tf.train.AdamOptimizer(learning_rate=lr)

        train_inp, train_gt, train_wt = datagen.generate_batch(batch_size=batch_size, nStacks=model.nStacks)
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
          [train_inp, train_gt, train_wt], capacity=2 * num_gpus)
        tower_grads = []
        
        print("Building Model")
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                        
                        image_batch, gt_batch, wt_batch = batch_queue.dequeue()
                        
                        loss = tower_loss(scope, image_batch, gt_batch, wt_batch)

                        tf.get_variable_scope().reuse_variables()
                        
                        model.training = False
                        
                        train_eval_output = model.inference(image_batch)

                        model.training = True

                        grads = optim.compute_gradients(loss)

                        tower_grads.append(grads)


        grads = average_gradients(tower_grads)

        apply_gradient_op = optim.apply_gradients(grads, global_step=global_step)

        variable_averages = tf.train.ExponentialMovingAverage(
            0.9999, global_step)
            
        with tf.control_dependencies([apply_gradient_op]):
            variables_averages_op   = variable_averages.apply(tf.trainable_variables())

        train_op = tf.group(apply_gradient_op, variables_averages_op)
        print("Model built in %.3f sec" % (time.time() - total_start))

        

        init = tf.global_variables_initializer()
        sess = tf.Session()
        saver = tf.train.Saver()
        start = time.time()
        if is_restore is True:
            saver.restore(sess, model.save_dir)
            tf.get_variable_scope().reuse_variables()
            model.epoch = int(sess.run(global_step) // epoch_size)
        else:
            print("\nInitializing Variables")
            
            sess.run(init)

        tf.train.start_queue_runners(sess=sess)
        print("Variables Initialized in %.3f sec" % (time.time() - start))
        for epoch in range(nEpochs):
            epoch_start_time = time.time()
            total_loss = 0
            accuracy = 0
            print("\nEpoch", model.epoch+1)
            for iteration in range(epoch_size):
                progBar(iteration+1, epoch_size)
                _, logits , gt,  stLoss = sess.run([train_op, train_eval_output, gt_batch, loss])
                total_loss += stLoss
                accur_pred = total_accuracy(logits, gt, nKeypoints=model.nKeypoints, nStacks=model.nStacks, batch_size=batch_size)
                # print(accur_pred)
                accuracy += np.sum(accur_pred)*100 / len(accur_pred)
            epoch_time = time.time() - epoch_start_time
            model.epoch += 1
            print('\nTrain Accuracy : %.3f' % (accuracy/epoch_size), '%')
            print("Loss : %f" % (total_loss/epoch_size), " Time Elapsed: %.3f" % epoch_time,"sec\n")
            
            if (epoch+1) % 5 == 0 or (epoch+1)==nEpochs:
                saver.save(sess, model.save_dir)
                print("Model Saved")

        print("Total Time Elapsed:  %.3f" % (time.time()-total_start),"sec")

train(nEpochs=50, learning_rate=2.5e-4, opt='adam', epoch_size=80, 
        batch_size=4, is_restore=False)

