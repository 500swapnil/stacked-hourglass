'''
Architecture adapted from https://github.com/bearpaw/pytorch-pose
'''
import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, MaxPool2D
from .residual import ResidualBlock

class Hourglass(tf.keras.Model):
    def __init__(self, n_blocks, filters, depth, name='hourglass_1'):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.upsample = UpSampling2D()
        self.hg = self.hourglass(n_blocks, filters, depth, name=name+'_depth_{0}'.format(depth))

    def residual_blocks(self, n_blocks, filters, name):
        layers = []
        for i in range(0, n_blocks):
            layers.append(ResidualBlock(filters))
        return tf.keras.Sequential(*layers, name=name)

    def hourglass(self, n_blocks, filters, depth, name='hourglass_1_1'):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self.residual_blocks(n_blocks, filters, name+'_res_{}'.format(j)))
            if i == 0:
                res.append(self.residual_blocks(n_blocks, filters, name+'_res_{}'.format(0)))
            hg.append(res)
        return hg

    def hourglass_call(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self.hourglass_call(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = self.upsample(low3)
        out = up1 + up2
        return out

    def call(self, x):
        return self.hourglass_call(self.depth, x)