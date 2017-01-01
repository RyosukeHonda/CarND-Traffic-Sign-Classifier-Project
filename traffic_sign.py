# Load pickled data
import pickle
import numpy as np
# TODO: fill this in based on where you saved the training and testing data
training_file = "traffic-signs-data/train.p"
testing_file = "traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

with open('additional_features.pickle', mode='rb') as f:
     train_features = pickle.load(f)

with open('additional_labels.pickle', mode='rb') as f:
     train_labels = pickle.load(f)

### To start off let's do a basic data summary.

# TODO: number of training examples
n_train = X_train.shape[0]

# TODO: number of testing examples
n_test = X_test.shape[0]

# TODO: what's the shape of an image?
image_shape = X_train[0,:,:,:].shape

# TODO: how many classes are in the dataset
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Preprocess the data here.
### Feel free to use as many code cells as needed.
X_train =X_train/255.0
#X_train = (X_train-X_train.mean())/X_train.std()
X_test =X_test/255.0
#X_test= (X_test-X_test.mean())/X_test.std()
### Generate data additional (if you want to!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.

#Split the data into train and validation data
from sklearn.cross_validation import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

import tensorflow as tf

from tensorflow.contrib.layers import flatten

def LeNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

features = tf.placeholder(tf.float32,shape=[None,32,32,3])
#labels = tf.placeholder(tf.int32,(None,43))
labels =tf.placeholder(tf.int32)
labels_oh= tf.one_hot(labels,43)
keep_prob = tf.placeholder("float")
print("Here")
#logits= structure_1(features)
logits = LeNet(features)
#cross_entropy=-tf.reduce_sum(labels*tf.log(logits))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels_oh)
loss_op= tf.reduce_mean(cross_entropy)
train_op= tf.train.AdamOptimizer().minimize(loss_op)

#preds = tf.arg_max(output,1)
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels_oh,1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

BATCH_SIZE=64
def eval_on_data(X,y):
    num_examples = len(X)
    total_acc = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0,num_examples,BATCH_SIZE):
        #X,y =shuffle(X,y)

        end = offset + BATCH_SIZE
        X_batch = X[offset:end]
        y_batch = y[offset:end]
        loss,acc = sess.run([loss_op,accuracy_op], feed_dict ={features:X_batch,labels:y_batch,keep_prob:1.0})
        total_loss += (loss*len(X_batch))
        total_acc += (acc*len(X_batch))
    return total_loss/num_examples,total_acc/num_examples


from sklearn.utils import shuffle
import time
training_epochs=100
batch_size=64
val_loss_his=[]
val_acc_his=[]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples=len(X_train)
    for i in range(training_epochs):
        X_train,y_train =shuffle(X_train,y_train)
        t0 =time.time()
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            X_batch, y_batch = X_train[offset:end], y_train[offset:end]
            sess.run(train_op, feed_dict={features: X_batch, labels: y_batch,keep_prob:0.5})

        val_loss, val_acc = eval_on_data(X_val, y_val)
        val_loss_his.append(val_loss)
        val_acc_his.append(val_acc)
        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss =%.4f" %(val_loss))
        print("Validation Accuracy =%.4f" %(val_acc))
        print("")

    try:
        saver
    except NameError:
        saver = tf.train.Saver()
        saver.save(sess, 'traffic_sign_keep0.5')
        print("Model saved")


import tensorflow as tf
with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    loader = tf.train.import_meta_graph('traffic_sign_keep0.5.meta')
    loader.restore(sess, tf.train.latest_checkpoint('./'))

    test_loss, test_acc = eval_on_data(X_test, y_test)
    print("Test Accuracy = {:.4f}".format(test_acc))