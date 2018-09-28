import tensorflow as tf
import numpy as np
        
input_dim=(64, 64)
nStacks=8
depth=4
nFeats=256
nModules=1
nKeypoints=10
save_dir='./saved_model/'
training=True
weight_decay=0.004
epoch = 0

def variable_on_cpu(name, shape, initializer, dtype=tf.float32, wd=weight_decay):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_decay')
            tf.add_to_collection('losses', weight_decay)
    return var

def conv2d(name, inputs, filters, kernel_size, stride=1,
            padd='VALID', act_relu=True, batch_norm=True):
    with tf.variable_scope(name) as scope:
        std = 1/np.sqrt(kernel_size*kernel_size*int(inputs.shape[3]))

        kernel = variable_on_cpu('kernel',shape=[kernel_size, kernel_size, int(inputs.shape[3]), int(filters)],
                    initializer=tf.random_uniform_initializer(-std, std))

        conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padd)

        bias = variable_on_cpu('bias',shape=[filters],initializer=tf.random_uniform_initializer(-std, std))

        conv = tf.nn.bias_add(conv, bias)
        
        if batch_norm:
            conv = tf.layers.batch_normalization(conv, name='bn', momentum=0.1, epsilon=1e-5 , training=training, trainable=True, 
                gamma_initializer=tf.random_uniform_initializer(0, 1))
        if act_relu:
            conv = tf.nn.relu(conv)
    return conv

def linear(name, inputs, num_out):
    return conv2d(name, inputs, filters=num_out, kernel_size=1, stride=1)


def skip_layer(name, inputs, num_out):
    if int(inputs.shape[3]) == num_out:
        return tf.identity(inputs, name='identity')
    else:
        return conv2d(name, inputs, filters=num_out, kernel_size=1, act_relu=False, batch_norm=False)

def conv_block(name, inputs, num_out):
    with tf.variable_scope(name) as scope:
        bn = tf.layers.batch_normalization(inputs, name='bn', momentum=0.1, epsilon=1e-5 ,training=training, trainable=True, 
            gamma_initializer=tf.random_uniform_initializer(0, 1))
        relu = tf.nn.relu(bn)
        conv1 = conv2d('conv1', relu, filters=num_out / 2, kernel_size=1)
        conv2 = conv2d('conv2', conv1, filters=num_out / 2,
                            kernel_size=3, padd="SAME")
        conv3 = conv2d('conv3', conv2, filters=num_out,
                            kernel_size=1, act_relu=False, batch_norm=False)
    return conv3

def residual(name, inputs, num_out):
    with tf.variable_scope(name) as scope:
        convB = conv_block('convB',inputs, num_out)
        skip = skip_layer('skip',inputs, num_out)
        convB = tf.add_n([convB, skip])
    return convB

def max_pool(inputs, kernel_size=2, stride=2, padd="VALID"):
    return tf.nn.max_pool(inputs, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padd)

def hourglass(name, inputs, n, num_out):

    with tf.variable_scope(name) as scope:
        up1 = inputs
        for i in range(nModules):
            up1 = residual('up1_%d'%i, up1, num_out)

        low1 = max_pool(inputs, kernel_size=2, stride=2)
        for i in range(nModules):
            low1 = residual('low1_%d'%i, low1, num_out)

        if n > 1:
            low2 = hourglass('low2', low1, n - 1, num_out)
        else:
            low2 = low1
            for i in range(nModules):
                low2 = residual('low2_%d'%i, low2, num_out)
        low3 = low2
        for i in range(nModules):
            low3 = residual('low3_%d'%i, low3, num_out)
        
        up2 = tf.image.resize_nearest_neighbor(low3, [int(low3.shape[1]) * 2, int(low3.shape[2]) * 2])
    return tf.add_n([up1, up2])

    
def inference(inputs):
    with tf.variable_scope('preprocessing') as scope:
        conv1 = conv2d('conv1', inputs, filters=64,
                            kernel_size=3, stride=1, padd="SAME")
        res1 = residual('res1', conv1, 128)
        # pool = max_pool(res1)
        res2 = residual('res2', res1, 128)
        res3 = residual('res3', res2, nFeats)

    
    out = [None for i in range(nStacks)]
    
    
    inter = res3

    for i in range(nStacks):
        with tf.variable_scope('stack_%d' % i) as scope:
            hg = hourglass('hg', inter, n=depth, num_out=nFeats)
            res = hg
            for j in range(nModules):
                res = residual('res_%d'%j,res, nFeats)

            lin = linear('lin',res, num_out=nFeats)

            out[i] = conv2d('out',lin, filters=nKeypoints,
                            kernel_size=1, stride=1, batch_norm=False, act_relu=False)

            if i < nStacks - 1:
                lin_ = conv2d('lin_',lin, filters=nFeats,
                            kernel_size=1, stride=1, batch_norm=False, act_relu=False)

                tmp_out = conv2d('tmp_out', out[i], filters=nFeats,
                            kernel_size=1, stride=1, batch_norm=False, act_relu=False)

                inter = tf.add_n([inter + lin_ + tmp_out])

    output = tf.stack(out, axis=1)
    return output

def loss(predictions, gt_map):
    losses = tf.reduce_mean(tf.losses.mean_squared_error(
        predictions=predictions, labels=gt_map))
    tf.add_to_collection('losses', losses)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

    
