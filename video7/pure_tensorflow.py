import numpy as np
import tensorflow as tf


# check if using GPU
print(tf.test.gpu_device_name(), end='\n\n')

t = tf.constant([1, 2, 3, 4, 5], dtype='float32')
x = np.array([5, 4, 3, 2, 1], dtype='float32')
t2 = tf.convert_to_tensor(x)
t3 = tf.square(t) + t2
print(t)
print(t2)
print(t3)
print(t.numpy())
print(t.device)

print(t.shape, t.dtype)

print(t[0], t2[:-2])
for e in t:
    tf.print('', e)

print(tf.random.uniform((3, 3)))


# Data
inputs = tf.convert_to_tensor(np.array([[0, 1], [1, 0], [0, 0], [1, 1]], dtype='float32'))
outputs = tf.convert_to_tensor(np.array([[1], [1], [0], [0]], dtype='float32'))

# Create model
hidden_size = 10
a_shape = inputs.shape
b_shape = (a_shape[1], hidden_size)
d_shape = outputs.shape
c_shape = (hidden_size, d_shape[1])
weights_b = tf.Variable(tf.random.normal(b_shape))
weights_c = tf.Variable(tf.random.normal(c_shape))
bias_b = tf.Variable(tf.zeros((1, b_shape[1])))
bias_c = tf.Variable(tf.zeros((1, c_shape[1])))
weights = [weights_b, bias_b, weights_c, bias_c]

# Loss and Optimizer
# optimizer = tf.optimizers.Adam() # Uses Keras

def sgd(weights, grads, lr=1.0):
    for ndx in range(len(grads)):
        weights[ndx].assign_sub(lr * grads[ndx])

def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def forward(x):
    x = tf.nn.relu(tf.matmul(x, weights_b) + bias_b)
    x = tf.math.sigmoid(tf.matmul(x, weights_c) + bias_c)
    return x

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        preds = forward(x) 
        loss = mse(y, preds)
    grads = tape.gradient(loss, weights)
    sgd(weights, grads)
    return loss

for epoch in range(100):
    print(f'Epoch: {epoch+1}', end='')
    loss = train_step(inputs, outputs)
    print(f' - {loss}')
    
print(forward(inputs))