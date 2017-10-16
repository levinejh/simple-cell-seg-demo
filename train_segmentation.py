# train_segmentation.py
# The goal is to train a simple convolutional network to perform cell segmentation.
# The network needs to classify each pixel as either "cell", "edge", or "non-cell".

import numpy as np
import tensorflow as tf
import random

nepochs = 300000
batchSize = 256
patchSize = 15
interval_display = 50
interval_save = 10000

save_path = '/home/ec2-user/Segmentation/Weights/segmentation_experiment_01/'
mask_path = '/home/ec2-user/Segmentation/Masks/'
image_path = '/home/ec2-user/Segmentation/Images/'

im_path = image_path + 'phase_27.tif'
mask_cells_path = mask_path + 'image_027_mask_cells.npy'
mask_edges_path = mask_path + 'image_027_mask_edges.npy'
mask_other_path = mask_path + 'image_027_mask_other.npy'

training_accuracy = np.zeros((2,1))

im = plt.imread(im_path)
mask_cells = np.load(mask_cells_path)
mask_edges = np.load(mask_edges_path)
mask_other = np.load(mask_other_path)

inds_cells = np.ravel_multi_index(np.where(mask_cells > 0), mask_cells.shape)
inds_edges = np.ravel_multi_index(np.where(mask_edges > 0), mask_edges.shape)
inds_other = np.ravel_multi_index(np.where(mask_other > 0), mask_other.shape)

def conv2d(x, W):
    #return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'VALID')

def maxpool2x2(x):
    #return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def getBatch(batchSize, patchSize, rawImage, ind_cells, ind_edges, ind_other):
    # load each batch in as a 1d array, then reshape
    # patchSize is the number of pixels around the pixel of interest
    # returns (imageBatch, labelBatch)
    imageBatch = np.zeros((batchSize, 2*patchSize+1, 2*patchSize+1, 1))
    labelBatch = np.zeros((batchSize,3))

    for i in range(batchSize):

        pixelType = random.randint(0,2)
        # 0: cells; 1: edges; 2: other.
        goodPixel = 0
        while (goodPixel == 0):
            if pixelType == 0:
                ind = ind_cells[random.randint(0, len(ind_cells)-1)]
            elif pixelType == 1:
                ind = ind_edges[random.randint(0, len(ind_edges)-1)]
            elif pixelType == 2:
                ind = ind_other[random.randint(0, len(ind_other)-1)]

            indX = np.unravel_index(ind, rawImage.shape)[1]
            indY = np.unravel_index(ind, rawImage.shape)[0]

            #ignore boundary pixels that need zero padding, ~5% of all pixels in image
            if ((indX - patchSize >= 0) and (indX + patchSize <= rawImage.shape[1]-1) and (indY - patchSize >= 0) and (indY + patchSize <= rawImage.shape[0]-1)):
                goodPixel = 1

        trainImage = rawImage[indY-patchSize : indY+patchSize+1, indX-patchSize : indX+patchSize+1]
        imageBatch[i,:,:,0] = trainImage
        labelBatch[i,pixelType] = 1

    return (imageBatch, labelBatch)

def save_weights(file_header, weight_dict):
    for j in weight_dict.keys():
        file_name = file_header + j
        print file_name
        np.save(file_name, weight_dict[j])


# Define the network
x = tf.placeholder(tf.float32, [batchSize, 2*patchSize+1, 2*patchSize+1, 1])
y_ = tf.placeholder(tf.float32, [batchSize, 3])
W1 = weight_variable([4,4,1,20])
b1 = bias_variable([20])
W2 = weight_variable([3,3,20,40])
b2 = bias_variable([40])
W3 = weight_variable([3,3,40,80])
b3 = bias_variable([80])
W4 = weight_variable([3,3,80,120])
b4 = bias_variable([120])
W5 = weight_variable([3,3,120,240])
b5 = weight_variable([240])
Wfc = weight_variable([240,1000])
bfc = bias_variable([1000])
Wout = weight_variable([1000,3])
bout = bias_variable([3])

h1_conv = tf.nn.relu( conv2d(x, W1) + b1 )
h1_pool = maxpool2x2(h1_conv)

h2_conv = tf.nn.relu(conv2d(h1_pool, W2) + b2)
h3_conv = tf.nn.relu(conv2d(h2_conv, W3) + b3)
h3_pool = maxpool2x2(h3_conv)

h4_conv = tf.nn.relu(conv2d(h3_pool, W4) + b4)
h5_conv = tf.nn.relu(conv2d(h4_conv, W5) + b5)

h5_flat = tf.reshape(h5_conv, [-1, 240])
h_fc = tf.nn.relu( tf.matmul(h5_flat, Wfc) + bfc )

y_conv = tf.matmul(h_fc, Wout) + bout

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    for i in range(nepochs):
        batch = getBatch(batchSize, patchSize, im, inds_cells, inds_edges, inds_other)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

        if i%interval_display == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1]})
            print("Epoch %d, Accuracy %g"%(i, train_accuracy))
            training_accuracy = np.append(training_accuracy, np.array([[i],[train_accuracy]]))
        if i%interval_save == 0:
            weights = {
                'W1': sess.run(W1),
                'b1': sess.run(b1),
                'W2': sess.run(W2),
                'b2': sess.run(b2),
                'W3': sess.run(W3),
                'b3': sess.run(b3),
                'W4': sess.run(W4),
                'b4': sess.run(b4),
                'W5': sess.run(W5),
                'b5': sess.run(b5),
                'Wfc': sess.run(Wfc),
                'bfc': sess.run(bfc),
                'Wout': sess.run(Wout),
                'bout': sess.run(bout),
                }
            save_weights(save_path + 'epoch_' + str(i) + '_', weights)

    np.save(save_path + 'training_accuracy', training_accuracy)
