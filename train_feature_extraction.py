import numpy as np
import pickle
import time
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
X_train, X_valid, y_train, y_valid = train_test_split(orig_data['features'], orig_data['labels'], test_size=0.33, random_state=0)

#print("len(X_train):", len(X_train), "  len(y_train):", len(y_train))
#print("len(X_valid):", len(X_valid), "  len(y_valid):", len(y_valid))
assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))

# TODO: Define placeholders and resize operation.
features = tf.placeholder(tf.float32, (None, 32, 32, 3), name="features")
labels = tf.placeholder(tf.int64, (None), name="labels")
resized = tf.image.resize_images(features, (227, 227))

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

#logits = tf.matmul(fc7, fc8W) + fc8b
#probs = tf.nn.softmax(logits)
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
#one_hot_y = tf.one_hot(labels, 43)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = RATE)
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])

prediction = tf.arg_max(logits, 1)
accuracy_operation = tf.reduce_mean(tf.cast(tf.equal(prediction, labels), tf.float32))
#correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
#accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#def evaluate(X_data, y_data):
#    num_examples = len(X_data)
#    total_accuracy = 0
#    sess = tf.get_default_session()
#    for offset in range(0, num_examples, BATCH_SIZE):
#        end_of_batch = offset+BATCH_SIZE
#        batch_x = X_data[offset:end_of_batch]
#        batch_y = y_data[offset:end_of_batch]
#        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
#        total_accuracy += (accuracy * len(batch_x))
#    return total_accuracy / num_examples

def eval_on_data(X, y, sess):
    total_accuracy = 0
    total_loss = 0
    for offset in range(0, X.shape[0], BATCH_SIZE):
        end = offset + BATCH_SIZE
        X_batch = X[offset:end]
        y_batch = y[offset:end]
#        loss, acc = sess.run([loss_operation, accuracy_operation], feed_dict={x: X_batch, y: y_batch})
#        loss = sess.run(loss_operation, feed_dict={x: X_batch, y: y_batch})
        loss, accuracy = sess.run([loss_operation, accuracy_operation], feed_dict={features: X_batch, labels: y_batch})
        total_loss += (loss * X_batch.shape[0])
        total_accuracy += (accuracy * X_batch.shape[0])
    return total_loss / X.shape[0], total_accuracy/X.shape[0]

# TODO: Train and evaluate the feature extraction model.
accuracy_train = np.zeros(EPOCHS)
accuracy_valid = np.zeros(EPOCHS)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    num_examples = len(X_train)

    print("Training...")
    for epoch in range(EPOCHS):
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={features: batch_x, labels: batch_y})

        valid_loss, valid_accuracy = eval_on_data(X_valid, y_valid, sess)
        print("EPOCH [{}] ... Time: {} seconds, validation loss: {:.3f}  accuracy: {:.3f}".format((epoch+1), (time.time()-t0), valid_loss, valid_accuracy))

#        accuracy_train[i] = evaluate(X_train, y_train)
#        accuracy_valid[i] = evaluate(X_valid, y_valid)
#        print("EPOCH [{}] ... Time: {:.3f} seconds, accuracy train: {:.3f} valid: {:.3f}".format(time.time() - t0, i+1, accuracy_train[i], accuracy_valid[i]))

