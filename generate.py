''' 
Own script to implement the CPPN image generator for understanding
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# definitions (defaults from original code)
z_dim = 16 
c_dim = 1
x_dim = 1080 
y_dim = 1060
n_pts = x_dim * y_dim
batch_size = 1
h_dim = 32 # hidden vector dimension
scale = 4.
n_layers = 3

# generate x, y, z
def coordinate_vecs(x_dim, y_dim, scale=1.0):
  x_half = (x_dim-1.) / 2.
  x_range = (np.arange(x_dim, dtype=np.float32) - x_half) / x_half
  # x_range = x_range * x_range
  x_range *= scale
  y_half = (y_dim-1.) / 2.
  y_range = (np.arange(y_dim, dtype=np.float32) - y_half) / y_half
  # y_range = y_range * y_range
  y_range *= scale

  x_mat = np.tile(x_range, y_dim).reshape(batch_size, -1, 1)
  y_mat = np.tile(y_range, (x_dim, 1)).reshape(batch_size, -1, 1, order='F')
  # original r_mat
  r_sqd = (x_mat * x_mat) + (y_mat * y_mat)
  r_mat = np.sqrt(r_sqd)

  return x_mat, y_mat, r_mat

def fc_layer(input_vec, hidden_size, with_bias = True):
  input_size = input_vec.get_shape().as_list()[1]

  weights = tf.Variable(
    # tf.truncated_normal([input_size, hidden_size],
    # stddev=1.0 / math.sqrt(float(input_size))))
    tf.random_normal([input_size, hidden_size],
    stddev=1.0)) 

  output = None
  if(with_bias):
    biases = tf.Variable(tf.random_normal([hidden_size], stddev=1.0))
    output = tf.matmul(input_vec, weights) + biases
  else:
    output = tf.matmul(input_vec, weights)

  return output

# model definition
z_ = tf.placeholder(tf.float32, [batch_size, z_dim], 'z_in')
x_ = tf.placeholder(tf.float32, [batch_size, None, 1], 'x_in')
y_ = tf.placeholder(tf.float32, [batch_size, None, 1], 'y_in')
r_ = tf.placeholder(tf.float32, [batch_size, None, 1], 'r_in')

z_mat = tf.reshape(
  tf.reshape(z_, [batch_size, 1, z_dim]) * 
    tf.ones([n_pts, 1], dtype=tf.float32),
  [batch_size * n_pts, z_dim]) * scale
x_flat = tf.reshape(x_, [-1, 1])
y_flat = tf.reshape(y_, [-1, 1])
r_flat = tf.reshape(r_, [-1, 1])

U = fc_layer(z_mat, h_dim, with_bias = False) + \
    fc_layer(x_flat, h_dim, with_bias = False) + \
    fc_layer(y_flat, h_dim, with_bias = False) + \
    fc_layer(r_flat, h_dim, with_bias = False)

# H = tf.nn.sigmoid(U)
H = tf.nn.tanh(U)
for i in range(n_layers):
  H = H + tf.nn.tanh(fc_layer(H, h_dim))

Hout = tf.nn.sigmoid(fc_layer(H, c_dim))

result = tf.reshape(Hout, [batch_size, y_dim, x_dim, c_dim])

if __name__=='__main__':
  # tensorflow commands to run coordinates through neural net
  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)

  # generate coordinates
  x_vec, y_vec, r_vec = coordinate_vecs(x_dim, y_dim, scale)
  # generate random latent vector z
  # z_vec = np.random.normal(loc=0.5, scale=1.2, size=(z_dim)).reshape(-1, z_dim)
  z_vec = np.random.uniform(-1.0, 1.0, z_dim).reshape(-1, z_dim)

  img_data = sess.run(result, feed_dict={x_ : x_vec, y_ : y_vec,
    z_ : z_vec, r_ : r_vec})[0]

  # print("Final image shape: {}".format(img_data.shape))

  plt.subplot(1, 1, 1)
  plt.imshow(img_data.squeeze(), cmap='Greys', interpolation='nearest')
  plt.axis('off')
  plt.show()
