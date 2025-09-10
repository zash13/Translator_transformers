from transformerBlock import TransformerBlock
from keras import layers


# The EncoderBlock has the same idea as the block I created in TransformerBlock.
# To clarify, I renamed it so that it matches the terminology used in the articles.
class EncoderBlock(layers.Layer):
    def __init__(self, embedding_dim, num_heads, feadForward_dim, rate=0.1):
        super().__init__()

        self.encoderBlock = TransformerBlock(
            embedding_dim, num_heads, feadForward_dim, rate=rate
        )

    def call(self, input):
        return self.encoderBlock.call(input)
