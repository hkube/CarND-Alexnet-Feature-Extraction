import numpy as np
import pickle
import tensorflow as tf
import sklearn.utils
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

nb_classes = 43

EPOCHS = 10
BATCH_SIZE = 128
RATE = 0.0003

# TODO: Load traffic signs data.
with open('train.p', mode='rb') as f:
	orig_data = pickle.load(f)

# TODO: Split data into training and validation sets.
X_orig, y_orig = orig_data['features'], orig_data['labels']
#print("len(X_orig):", len(X_orig), "  len(y_orig):", len(y_orig))
assert(len(X_orig) == len(y_orig))

# Use 1/10th of the data for validation
validataionSetSize = len(X_orig) // 10
X_train = X_orig[:validataionSetSize]
y_train = y_orig[:validataionSetSize:]
X_valid = X_orig[validataionSetSize:]
y_valid = y_orig[validataionSetSize:]

#print("len(X_train):", len(X_train), "  len(y_train):", len(y_train))
#print("len(X_valid):", len(X_valid), "  len(y_valid):", len(y_valid))
assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix

# fc8, 1000
fc8W = tf.Variable(tf.random_normal(shape, mean=0, stddev=0.1), name="fc8W")
fc8b = tf.Variable(tf.zeros(nb_classes), name="fc8b")

logits = tf.matmul(fc7, fc8W) + fc8b
probs = tf.nn.softmax(logits)


# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
y = tf.placeholder(tf.int32, (None), name="y")
one_hot_y = tf.one_hot(y, 43)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = RATE)
training_operation = optimizer.minimize(loss_operation)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# TODO: Train and evaluate the feature extraction model.
accuracy_train = np.zeros(EPOCHS)
accuracy_valid = np.zeros(EPOCHS)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    num_examples = len(X_train)
    
    print("Training...")
    for i in range(EPOCHS):
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        accuracy_train[i] = evaluate(X_train, y_train)
        accuracy_valid[i] = evaluate(X_valid, y_valid)
        print("EPOCH [{}] ... accuracy train: {:.3f} valid: {:.3f}".format(i+1, accuracy_train[i], accuracy_valid[i]))

