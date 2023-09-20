import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define the generator network
generator = keras.Sequential([
    layers.Dense(256, input_shape=(100,), activation='relu'),
    layers.Dense(784, activation='tanh')
])

# Define the discriminator network
discriminator = keras.Sequential([
    layers.Dense(256, input_shape=(784,), activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Define loss function and optimizers
criterion = keras.losses.BinaryCrossentropy()
optimizer_g = keras.optimizers.Adam(learning_rate=0.0002)
optimizer_d = keras.optimizers.Adam(learning_rate=0.0002)

# Load the MNIST dataset
mnist = keras.datasets.mnist
(train_images, _), (_, _) = mnist.load_data()
train_images = train_images / 255.0
train_images = train_images.reshape(train_images.shape[0], 784)

# Training loop
num_epochs = 100
batch_size = 64
for epoch in range(num_epochs):
    for batch_start in range(0, train_images.shape[0], batch_size):
        batch_end = batch_start + batch_size
        real_images = train_images[batch_start:batch_end]

        # Train the discriminator
        with tf.GradientTape() as tape_d:
            real_labels = tf.ones((batch_size, 1))
            fake_labels = tf.zeros((batch_size, 1))

            # Discriminator loss for real images
            real_outputs = discriminator(real_images)
            d_loss_real = criterion(real_labels, real_outputs)

            # Generate fake images and compute discriminator loss for fake images
            z = tf.random.normal((batch_size, 100))
            fake_images = generator(z)
            fake_outputs = discriminator(fake_images)
            d_loss_fake = criterion(fake_labels, fake_outputs)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake

        grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
        optimizer_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as tape_g:
            fake_outputs = discriminator(fake_images)
            g_loss = criterion(real_labels, fake_outputs)

        grads_g = tape_g.gradient(g_loss, generator.trainable_variables)
        optimizer_g.apply_gradients(zip(grads_g, generator.trainable_variables))

    print(f"Epoch [{epoch}/{num_epochs}] D Loss: {d_loss.numpy():.4f} G Loss: {g_loss.numpy():.4f}")

    # Save generated images at the end of each epoch
    if (epoch + 1) % 10 == 0:
        n = 10  # Generate 10x10 grid of images
        generated_images = generator(tf.random.normal((n * n, 100)))
        generated_images = generated_images.numpy().reshape(-1, 28, 28)

        plt.figure(figsize=(10, 10))
        for i in range(n * n):
            plt.subplot(n, n, i + 1)
            plt.imshow(generated_images[i], cmap='gray')
            plt.axis('off')
        plt.savefig(f'gan_generated_epoch_{epoch + 1}.png')
        plt.close()
