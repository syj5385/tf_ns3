import tensorflow as tf

input_size = 2
output_size = 3
numOfHidden = 10

layer = (50, 40, 30)

W = list()
#W.extend([1,len(layer)+2])
b = list()
#b.extend([1,len(layer)+2])
h_layer = list()
#h_layer.extend([1,len(layer)+2])


# Batch of input and target output (1x1 matrices)

# Set input
x = tf.placeholder(tf.float32, shape=[None,  input_size], name='input')
y = tf.placeholder(tf.float32, shape=[None,  output_size], name='target')

print (layer[0])
# configure input layer
W.append(tf.get_variable("W0", shape=[input_size, layer[0]], initializer = tf.contrib.layers.xavier_initializer()))
b.append(tf.get_variable("b0", shape=[layer[0]], initializer = tf.contrib.layers.xavier_initializer()))
h_layer.append(tf.nn.tanh(tf.matmul(x,W[0]) + b[0]))

for i in range(len(layer)-1):
	w_t = str("W{}".format(i+1))
	b_t = str("b{}".format(i+1))
	W.append(tf.get_variable(w_t, shape=[layer[i], layer[i+1]], initializer =tf.contrib.layers.xavier_initializer()))
	b.append(tf.get_variable(b_t, shape=[layer[i+1]], initializer = tf.contrib.layers.xavier_initializer()))
	h_layer.append(tf.nn.tanh(tf.matmul(h_layer[i],W[i+1]) + b[i+1]))

w_t = "W{}".format(len(layer) + 1)
b_t = "b{}".format(len(layer) + 1)

o_W = tf.get_variable("o_W", shape=[layer[len(layer)-1], output_size], initializer = tf.contrib.layers.xavier_initializer())
o_b = tf.get_variable("o_b", shape=[output_size], initializer = tf.contrib.layers.xavier_initializer())

y_ = tf.identity(tf.matmul(h_layer[i+1], o_W) + o_b, name='output')

# Trivial linear model
#y_ = tf.identity(tf.layers.dense(x, 1), name='output')


# Optimize loss
loss = tf.reduce_mean(tf.square(y_ - y), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss, name='train')

init = tf.global_variables_initializer()

# tf.train.Saver.__init__ adds operations to the graph to save
# and restore variables.
saver_def = tf.train.Saver().as_saver_def()

print('Run this operation to initialize variables     : ', init.name)
print('Run this operation for a train step            : ', train_op.name)
print('Feed this tensor to set the checkpoint filename: ', saver_def.filename_tensor_name)
print('Run this operation to save a checkpoint        : ', saver_def.save_tensor_name)
print('Run this operation to restore a checkpoint     : ', saver_def.restore_op_name)

# Write the graph out to a file.
with open('graph.pb', 'w') as f:
  f.write(tf.get_default_graph().as_graph_def().SerializeToString())
