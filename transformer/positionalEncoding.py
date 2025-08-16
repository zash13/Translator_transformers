import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# This code is adapted from https://www.geeksforgeeks.org/nlp/positional-encoding-in-transformers/
# The article also provides a detailed explanation of how it works.


def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(
        10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model)
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


position = 10
d_model = 8
pos_encoding = positional_encoding(position, d_model)

print("Shape:", pos_encoding.shape)
print("Values:\n", pos_encoding.numpy()[0, :, :])

plt.matshow(pos_encoding[0].numpy())
plt.colorbar()
plt.title("Positional Encoding (pos=10, d_model=8)")
plt.show()
