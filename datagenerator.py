import os
import cv2
import pandas as pd
import numpy as np


class DataGenerator():

    def __init__(self, img_dir=None, nKeypoints=10, data_file=None, dim=(64, 64)):
        self.nKeypoints = nKeypoints
        self.img_dir = img_dir
        self.data_file = data_file
        self.width = dim[0]
        self.height = dim[1]
        self.table = pd.read_csv(data_file, header=None, delimiter=',', index_col=0).to_dict('index')
        # self.table = df[(df[df.columns] != -1).all(axis=1)].to_dict('index')

    def gaussian(self, s, center):
        x,y = np.meshgrid(np.linspace(0,64,64),np.linspace(0,64,64))    
        c_x, c_y = center
        domain_kernel = np.exp(-((x-c_x)**2 + (y-c_y)**2)/(2*(1**2)))
        return domain_kernel

    def gen_heatmap(self, keypoints):

        hm = np.zeros([self.width, self.height, self.nKeypoints])
        wt = np.ones_like(hm)
        for i in range(self.nKeypoints):
            keypointX, keypointY = keypoints[2*i:2*i+2]
            if keypointX != -1 and keypointY != -1:
                hm[:, :, i] = self.gaussian(s=3, center=(keypointY, keypointX))
            # else:
            #     wt[:, :, i] = np.zeros([self.width, self.height])
        return hm, wt

    def read_image(self, loc):
        return cv2.imread(loc)/255

    def generate_batch(self, batch_size=16, nStacks=2):

        train_inp = np.zeros(
            (batch_size, self.width, self.height, 3), dtype=np.float32)
        train_gt = np.zeros((batch_size, nStacks, self.width,
                             self.height, self.nKeypoints), dtype=np.float32)
        train_wt = np.ones_like(train_gt)
        cur_batch = np.random.choice(list(self.table.keys()), batch_size)
        # cur_batch = [str(i) + '.jpg' for i in range(1,17)]
        for i in range(batch_size):
            train_inp[i] = self.read_image(os.path.join(self.img_dir, cur_batch[i]))
            keypoints = list(self.table[cur_batch[i]].values())
            heatmaps, weights = self.gen_heatmap(keypoints)
            heatmaps = np.expand_dims(heatmaps, axis=0)
            weights = np.expand_dims(weights, axis=0)
            train_gt[i] = np.repeat(heatmaps, nStacks, axis=0)
            train_wt[i] = np.repeat(weights, nStacks, axis=0)

        return train_inp, train_gt, train_wt
