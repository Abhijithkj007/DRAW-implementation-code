#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import the necessary libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the DRAW model class
class DRAWModel(tf.keras.Model):
    def __init__(self, img_size, num_steps, batch_size):
        super(DRAWModel, self).__init__()
        self.img_size = img_size
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.build_model()

    def build_model(self):
        # Define the LSTM cell for the encoder and decoder
        self.lstm_enc = tf.keras.layers.LSTMCell(units=256)
        self.lstm_dec = tf.keras.layers.LSTMCell(units=256)

        # Define the learnable parameters for the model
        self.W_enc = tf.Variable(tf.random.normal([256, 10]), name='W_enc')
        self.b_enc = tf.Variable(tf.zeros([10]), name='b_enc')

        self.W_dec = tf.Variable(tf.random.normal([256, self.img_size * self.img_size]), name='W_dec')
        self.b_dec = tf.Variable(tf.zeros([self.img_size * self.img_size]), name='b_dec')

    def encode(self, enc_state, input):
        _, enc_state = self.lstm_enc(input, enc_state)
        return enc_state

    def decode(self, dec_state, input):
        _, dec_state = self.lstm_dec(input, dec_state)
        return dec_state

    def read(self, x, x_hat, h_dec_prev):
        # Define the parameters for the read operation
        kernel_size = 3
        stride = 2

        # Compute the read window using a convolutional filter
        read_window = tf.image.extract_patches(
            images=x_hat,
            sizes=[1, kernel_size, kernel_size, 1],
            strides=[1, stride, stride, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'
        )

        # Reshape the read window to match the LSTM input shape
        read_window = tf.reshape(read_window, [self.batch_size, -1])

        # Concatenate the read window with the previous decoder hidden state
        read_input = tf.concat([read_window, h_dec_prev], axis=1)


        return read_input

    def write(self, h_dec, canvas):
        # Map the hidden state to the canvas size
        write_output = tf.matmul(h_dec, self.W_dec) + self.b_dec

        # Reshape the write output to match the canvas shape
        write_output = tf.reshape(write_output, [self.batch_size, self.img_size, self.img_size])

        # Add the write output to the canvas using element-wise addition
        canvas += write_output

        return canvas

    def attention(self, h_dec, enc_state,x):
        # Compute glimpse window
        glimpse = self.read(x, h_dec)

        # Apply attention mechanism
        glimpse = tf.reshape(glimpse, [self.batch_size, -1])
        glimpse_attention = tf.nn.softmax(tf.matmul(glimpse, self.W_enc) + self.b_enc, axis=1)

        # Compute weighted sum of glimpse window
        glimpse_vector = tf.reduce_sum(glimpse * tf.expand_dims(glimpse_attention, axis=2), axis=1)

        return glimpse_vector

    def call(self, x):
    # Expand dimensions of input tensor
        x = tf.expand_dims(x, axis=-1)

    # Reshape input tensor to have 4 dimensions
        x = tf.reshape(x, [self.batch_size, -1, 1])

    # Define the initial states for the encoder and decoder LSTM
        self.enc_state = [tf.zeros([self.batch_size, 256]), tf.zeros([self.batch_size, 256])]
        self.dec_state = [tf.zeros([self.batch_size, 256]), tf.zeros([self.batch_size, 256])]

    # Define the initial canvas
        self.canvas = tf.zeros([self.batch_size, self.img_size, self.img_size, 1])

    # Define the recurrent steps of the DRAW model
        def draw_recurrent_step(t, x, h_dec_prev, enc_state, canvas):
        # Read operation
            x_hat = x - tf.sigmoid(canvas)
            read_window = self.read(x, x_hat, h_dec_prev)

        # Encoder operation
            r, enc_state = self.encode(enc_state, tf.concat([read_window, h_dec_prev], axis=1))

        # Attention operation
            z = self.attention(r, enc_state, x)  # Pass x to the attention method

        # Decoder operation
            h_dec, dec_state = self.decode(self.dec_state, z)

        # Write operation
            canvas = self.write(h_dec, canvas)
    
            return t + 1, x, h_dec, enc_state, canvas

    # Define the loop for the recurrent steps
        def draw_recurrence(t, x, h_dec_prev, enc_state, canvas):
            return tf.less(t, self.num_steps)

    # Execute the recurrent steps of the DRAW model
        _, _, _, _, final_canvas = tf.while_loop(
        cond=draw_recurrence,
        body=draw_recurrent_step,
        loop_vars=[0, x, self.dec_state, self.enc_state, self.canvas]
        )

        return final_canvas



# Set hyperparameters
img_size = 28
num_steps = 10
batch_size = 64
learning_rate = 0.001
epochs = 10

# Create an instance of the DRAW model
draw_model = DRAWModel(img_size, num_steps, batch_size)

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Define the loss function
loss_fn = tf.keras.losses.MeanSquaredError()

# Compile the model
draw_model.compile(optimizer=optimizer, loss=loss_fn)

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    num_batches = x_train.shape[0] // batch_size
    for batch in range(num_batches):
        
        x_batch = x_train[batch * batch_size: (batch + 1) * batch_size]
        with tf.GradientTape() as tape:
            outputs = draw_model(x_batch)
            loss_value = loss_fn(x_batch, outputs)
        gradients = tape.gradient(loss_value, draw_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, draw_model.trainable_variables))
        print(f"Batch {batch + 1}/{num_batches}, Loss: {loss_value:.4f}")

    # Evaluate the model on test data after each epoch
    test_outputs = draw_model(x_test)
    test_loss = loss_fn(x_test, test_outputs)
    print(f"Test Loss: {test_loss:.4f}")

print("Training completed.")

# Generate reconstructions on test data
reconstructions = draw_model(x_test).numpy()

# Display original images and their reconstructions
n = 10  # number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    plt.title("Original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructions[i])
    plt.title("Reconstruction")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




