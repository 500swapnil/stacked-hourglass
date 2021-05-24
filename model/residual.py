import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU

class ResidualBlock(tf.keras.Model):

    def __init__(self, planes, strides=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.bn1 = BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.conv1 = Conv2D(planes, kernel_size=1,)
        self.bn2 = BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.conv2 = Conv2D(planes, kernel_size=3, strides=strides,
                               padding="same",)
        self.bn3 = BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.conv3 = Conv2D(planes * 2, kernel_size=1,)
        self.relu = ReLU()
        self.downsample = downsample

    def call(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out