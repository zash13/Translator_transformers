from transformerBlock import TransformerBlock
from keras import layers


# The EncoderBlock has the same idea as the block I created in TransformerBlock.
# To clarify, I renamed it so that it matches the terminology used in the articles.


class EncoderBlock(layers.Layer):
    def __init__(self, embedding_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.encoderBlock = TransformerBlock(
            embedding_dim, num_heads, feed_forward_dim, rate=rate
        )

    def build(self, input_shape):
        super().build(input_shape)  # Ensure build is called

    def call(self, inputs, training=True, padding_mask=None):
        return self.encoderBlock(inputs, training=training, padding_mask=padding_mask)
