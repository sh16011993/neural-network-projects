# import necessary libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import datetime
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
from sklearn.model_selection import train_test_split


tf.logging.set_verbosity(tf.logging.INFO)

def load_mnist_data(dataset_type, folder_path):
  if dataset_type == "train":
    data_path = folder_path+"train-images.idx3-ubyte"
    labels_path = folder_path+"train-labels.idx1-ubyte"
    
  if dataset_type == "test":
    data_path = folder_path+"t10k-images.idx3-ubyte"
    labels_path = folder_path+"t10k-labels.idx1-ubyte"

  images, labels = loadlocal_mnist(
        images_path=data_path, 
        labels_path=labels_path)

  return images, labels


def cnn_model_mnist(features, labels, mode, params):
  # Hyper parameters to be tuned
  learning_rate = params["learning_rate"]
  dropout_rate = params["dropout_rate"]
  activation_fn = params["activation_function"]

  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST contains monochrome images of 28*28
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 28 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 28]
  convolution_1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=28,
      kernel_size=[5, 5],
      padding="same",
      activation=activation_fn)

  # Pooling Layer #1
  # Max pooling layer with a 2x2 filter and stride of 2
  # Results in downsampling by a factor 2
  # Input Tensor Shape: [batch_size, 28, 28, 28]
  # Output Tensor Shape: [batch_size, 14, 14, 28]
  pool_1 = tf.layers.max_pooling2d(inputs=convolution_1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 56 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 28]
  # Output Tensor Shape: [batch_size, 14, 14, 56]
  convolution_2 = tf.layers.conv2d(
      inputs=pool_1,
      filters=56,
      kernel_size=[5, 5],
      padding="same",
      activation=activation_fn)

  # Pooling Layer #2
  # Max pooling layer with a 2x2 filter and stride of 2
  # Results in downsampling by a factor 2
  # Input Tensor Shape: [batch_size, 14, 14, 56]
  # Output Tensor Shape: [batch_size, 7, 7, 56]
  pool_2 = tf.layers.max_pooling2d(inputs=convolution_2, pool_size=[2, 2], strides=2)

  # Unwrap tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 56]
  # Output Tensor Shape: [batch_size, 7 * 7 * 56]
  pool_2_unwrap = tf.reshape(pool_2, [-1, 7 * 7 * 56])

  # Dense Layer
  # Fully connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 56]
  # Output Tensor Shape: [batch_size, 1024]
  fully_connected_1 = tf.layers.dense(inputs=pool_2_unwrap, units=1024, activation=activation_fn)

  # Include dropout with 0.5 probability that element will be kept
  dropout_fc1 = tf.layers.dropout(
      inputs=fully_connected_1, rate=dropout_rate, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Output layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  output_layer = tf.layers.dense(inputs=dropout_fc1, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=output_layer, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(output_layer, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=output_layer)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(undef_args):
  # load training images and corresponding labels
  train_img, train_lbl = load_mnist_data("train","./dataset/")

  # load testing images and corresponding labels
  tst_img, tst_lbl = load_mnist_data("test","./dataset/")

  #normalize the data
  train_img = train_img/np.float32(255)
  train_lbl = train_lbl.astype(np.int32)
  tst_img = tst_img/np.float32(255)
  tst_lbl = tst_lbl.astype(np.int32)

  # split data into training and validation sets
  train_size = 0.7
  val_size = 0.3
  train_ph_img, val_img, train_ph_lbl, val_lbl = train_test_split(
  	train_img, train_lbl, train_size=train_size, test_size=val_size, random_state=0)

  # Hyper parameters to be tuned
  batch_size = 100
  num_epochs = 40
  learning_rate = 0.01
  dropout_rate = 0.4
  #num_epochs_range = [10, 20, 40, 80, 100, 200]
  #learning_rate_range = [0.001, 0.01, 0.1, 1.0]
  #dropout_rate_range = [0.2, 0.4, 0.6, 0.8]
  training_accuracy = []
  training_loss = []
  val_accuracy = []
  val_loss = []
  act_fns = [tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid]
  for act_fn in act_fns:
    params = {'learning_rate': learning_rate, 'dropout_rate': dropout_rate, 'activation_function': act_fn}
	# Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
	  model_fn=cnn_model_mnist, params = params)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
	  x={"x": train_ph_img},
	  y=train_ph_lbl,
	  batch_size=batch_size,
	  num_epochs=num_epochs,
	  shuffle=True)

    mnist_classifier.train(input_fn=train_input_fn)

    # Evaluate the model and store results on training data
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_ph_img}, y=train_ph_lbl, num_epochs=1, shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    training_accuracy.append(eval_results["accuracy"])
    training_loss.append(eval_results["loss"])

    # Evaluate the model and store results on validation data
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	  x={"x": val_img}, y=val_lbl, num_epochs=1, shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    val_accuracy.append(eval_results["accuracy"])
    val_loss.append(eval_results["loss"])

    # Convert data suitably
  val_accuracy_percent = [x*100 for x in val_accuracy]
  training_accuracy_percent = [x*100 for x in training_accuracy]
	
  activation_fns = ['Relu', 'Tanh', 'Sigmoid']

  print('Printing Validation Accuracy and Loss: ')
  for i in range(len(val_accuracy_percent)):
    print(val_accuracy_percent[i])
    print(val_loss[i])
  print('Printing Training Accuracy and Loss: ')
  for i in range(len(training_accuracy_percent)):
    print(training_accuracy_percent[i])
    print(training_loss[i])
		
  plt.figure()
  plt.bar(np.arange(len(activation_fns)), training_accuracy_percent, align='center', alpha=0.5)
  plt.xticks(np.arange(len(activation_fns)), activation_fns)
  plt.yticks(np.arange(0, 101, 5))
  plt.xlabel('Activation Functions')
  plt.ylabel('Training Accuracy %')
  plt.title('Plot of Activation Functions vs Training Accuracy')
  plt.savefig('./plot_act_fn_Train_Accuracy_1.png', format = 'png')
  
  plt.figure()
  plt.bar(np.arange(len(activation_fns)), val_accuracy_percent, align='center', alpha=0.5)
  plt.xticks(np.arange(len(activation_fns)), activation_fns)
  plt.yticks(np.arange(0, 101, 5))
  plt.xlabel('Activation Functions')
  plt.ylabel('Validation Accuracy %')
  plt.title('Plot of Activation Functions vs Validation Accuracy')
  plt.savefig('./plot_act_fn_Validation_Accuracy_1.png', format = 'png')
  
  plt.show()

  # # plot the learning curve
  # plt.figure()
  # plt.plot(dropout_rate_range, val_loss, 'r', label='validation loss')
  # plt.plot(dropout_rate_range, training_loss, 'g', label='training loss')
  # plt.xlabel('Dropout Rate')
  # plt.ylabel('loss')
  # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
  #          ncol=2, mode="expand", borderaxespad=0.)
  # plt.savefig('./plot_loss.png', format = 'png')
  #
  # plt.figure()
  # plt.plot(dropout_rate_range, val_accuracy_percent, 'b', label='validation accuracy')
  # plt.plot(dropout_rate_range, training_accuracy_percent, 'y', label='training accuracy')
  # plt.xlabel('Dropout Rate')
  # plt.ylabel('accuracy (%)')
  # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
  #          ncol=2, mode="expand", borderaxespad=0.)
  # plt.savefig('./plot_accuracy.png', format = 'png')
  # plt.show()


if __name__ == "__main__":
  tf.app.run()
