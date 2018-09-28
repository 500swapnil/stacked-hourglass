import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf

class DataGenerator():

    def __init__(self, img_dir=None, nKeypoints=10, data_file=None, nStacks=8, dim=(64, 64)):
        self.nKeypoints = nKeypoints
        self.img_dir = img_dir
        self.nStacks = nStacks
        self.data_file = data_file
        self.width = dim[0]
        self.height = dim[1]
        self.table = pd.read_csv(data_file, header=None, delimiter=',', index_col=None)
        self.table[0] = img_dir + '/' + self.table[0]
        self.table = self.table.set_index(0)
        # self.table = self.table.to_dict('index')
        # self.table = df[(df[df.columns] != -1).all(axis=1)].to_dict('index')

    def gaussian(self, s, center):
        x,y = tf.meshgrid(tf.linspace(0.0,64.0,64),tf.linspace(0.0,64.0,64))    
        c_x, c_y = center
        domain_kernel = tf.exp(-((x-c_x)**2 + (y-c_y)**2)/(2*(1**2)))
        return domain_kernel

    def gen_heatmap(self, keypoints):
        hm = []
        for i in range(self.nKeypoints):
            keypointX = keypoints[2*i]
            keypointY = keypoints[2*i+1]
            miss = tf.constant(-1.0,dtype=tf.float32)
            def f1(): 
                return self.gaussian(s=3, center=(keypointY, keypointX))
            def f2():
                return tf.zeros((self.width, self.height))
            hm.append(tf.cond(tf.equal(keypointX,miss),f2,f1))
        hm = tf.stack(hm,axis=2)
        return tf.expand_dims(hm,axis=0)

    def read_image(self, input_queue):
        keypoints = input_queue[1]
        contents = tf.read_file(input_queue[0])
        image = tf.image.decode_jpeg(contents,channels=3)
        image_resized = tf.image.resize_images(image, [self.width, self.height])
        gt_map = self.gen_heatmap(keypoints)
        gt_map = tf.tile(gt_map,[self.nStacks,1,1,1])
        return image_resized, gt_map

    def generate_image_and_label_batch(self, image, label, min_queue_examples,
                                    batch_size, shuffle):
        num_preprocess_threads = 16
        if shuffle:
            images, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples)
        else:
            images, label_batch = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size)

        return images, label_batch

    def input_pipeline(self, batch_size=8, num_examples_per_epoch=1000):
        filenames = tf.constant(list(self.table.index))
        keypoints = tf.constant(self.table.values,dtype=tf.float32)
        
        input_queue = tf.train.slice_input_producer([filenames, keypoints])
        image, label = self.read_image(input_queue)
        float_image = tf.cast(image, tf.float32)
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

        return self.generate_image_and_label_batch(float_image, label, min_queue_examples,
                                    batch_size,shuffle=False)