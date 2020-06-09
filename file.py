import tensorflow as tf

# check version

tf.logging.set_verbosity(tf.logging.ERROR)
print('Using TensorFlow version', tf.__version__)

# import dataset

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 60000 train
# 10000 test
# 28 by 28 images

# plot example

from matplotlib import pyplot as plot
plt.imshow(x_train[0], cmap = 'binary')
plt.show()

# one hot encoding

from tensorflow.keras.utils import to_categorical
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

