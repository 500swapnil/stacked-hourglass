from datagenerator import DataGenerator
import multi_hourglass as model
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

sess = tf.Session()

datagen = DataGenerator(img_dir='chair1', nKeypoints=10, data_file='chair_mini.txt')
model.nFeats = 256
model.nStacks = 8
model.training = True
batch_size = 4
def show_output():
    input_im, gt, wt = datagen.generate_batch(batch_size=batch_size,nStacks=model.nStacks)
    
    global_step = tf.get_variable('global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
    output = model.inference(input_im)
    
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True))
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, model.save_dir)
    tf.get_variable_scope().reuse_variables()
    hm = sess.run(output)
    # print(hm[0,model.nStacks-1,:,:,0])
    for k in range(batch_size):
        plt.imshow(input_im[k,:,:,:])
        for i in range(10):
            x, y = coord(hm[k,model.nStacks-1,:,:,i])
            # print(x,y)
            plt.scatter(x, y, s=10, c='red', marker='x')
        plt.savefig(str(k) + '_' + str(int(sess.run(global_step))) + '.png')
        print(k)

def coord(hm):
    return (np.argmax(hm) // datagen.width, np.argmax(hm) % datagen.width)

def mark_keypoints(image_name):
    image = datagen.read_image(image_name)
    keypoints = list(datagen.table[image_name[7:]].values())
    hm,wt = datagen.gen_heatmap(keypoints)
    k = []
    # print(hm.shape)
    plt.imshow(image)
    for i in range(10):
        x, y = coord(hm[:,:,i])
        plt.scatter(x, y, s=10, c='red', marker='x')
    plt.show()

show_output()