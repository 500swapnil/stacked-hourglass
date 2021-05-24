import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU
from model.residual import ResidualBlock
from model.hourglass import Hourglass

class StackedHourglass(tf.keras.Model):

    def __init__(self, n_stacks=2, n_blocks=1, n_keypoints=10):
        super(StackedHourglass, self).__init__()

        self.inplanes = 64
        self.expansion = 2
        self.num_feats = 128
        self.n_stacks = n_stacks
        self.conv1 = Conv2D(self.inplanes, kernel_size=3, padding='same')
        self.bn1 = BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.relu = ReLU()
        self.residual1 = self.residual_blocks(128, 1)
        self.residual2 = self.residual_blocks(128, 1)
        self.residual3 = self.residual_blocks(self.num_feats, 1)

        # build hourglass modules
        expanded_feats = self.num_feats*self.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(n_stacks):
            hg.append(Hourglass(n_blocks, self.num_feats, 4, 'hourglass_{}'.format(i)))
            res.append(self.residual_blocks(self.num_feats, n_blocks))
            fc.append(self.linear(expanded_feats))
            score.append(Conv2D(n_keypoints, kernel_size=1))
            if i < n_stacks-1:
                fc_.append(Conv2D(expanded_feats, kernel_size=1))
                score_.append(Conv2D(expanded_feats, kernel_size=1))
        self.hg = hg
        self.res = res
        self.fc = fc
        self.score = score
        self.fc_ = fc_
        self.score_ = score_

    def residual_blocks(self, planes, n_blocks, strides=1):
        downsample = None
        if strides != 1 or self.inplanes != planes * self.expansion:
            downsample = Conv2D(planes * self.expansion, kernel_size=1, strides=strides)
        layers = []
        layers.append(ResidualBlock(planes, strides, downsample))
        self.inplanes = planes * self.expansion
        for i in range(1, n_blocks):
            layers.append(ResidualBlock(planes))

        return tf.keras.Sequential(*layers)

    def linear(self, outplanes):
        bn = BatchNormalization(momentum=0.1, epsilon=1e-5)
        conv = Conv2D(outplanes, kernel_size=1)
        return tf.keras.Sequential([conv, bn, self.relu])

    def call(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 

        x = self.residual1(x)  
        x = self.residual2(x)  
        x = self.residual3(x)  

        for i in range(self.n_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.n_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out