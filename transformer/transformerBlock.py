import tensorflow as tf
import keras
from keras import layers
import numpy as np

# this is my first project on transformers , basicly its using others code to learn how thing work in here
# this is where you can find where the origin of the code come from :
# https://keras.io/examples/nlp/text_classification_with_transformer/


class TransformerBlock(layers.Layer):
    def __init__(self, embeding_dim, num_heads, feadForward_dim, rate=0.1):
        super().__init__()
        # its compute how much each token should pay attentoin to others
        self.attention_layer = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embeding_dim
        )

        # this layer is on each toekn , while attention mixes the information across tokens this layers help each token to have its own idea
        # like attention care about sequence , while each part of sequence need to attent to for it self
        # if i want to put it in an example , its like every toeken know the idea of the sequence , but each one hase its own thouts too!!!
        # like people in same party  , which all know the general idea of the party , but each one hase its own thinking too( if so :) )
        self.feedforward_network = keras.Sequential(
            [
                layers.Dense(feadForward_dim, activation="relu"),
                layers.Dense(embeding_dim),
            ]
        )
        # still unfamilier with normalizatoin , but i know it for facat that Dropout is going to to Dropout some neurons :))))) like turning them off !
        # and preventing from overfitting
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def feedforward(self, inputs):
        # if you not get this part to , read articles about differece between self attention and attention( i think so )
        # i think in self attention , each token look at all tokens including itself , so query = key = value = input
        # while in attention it self , we have Q and V and K
        # like in cross-attention (better name then attention is self) we have Q that comes from oen source while K and V comes from another
        # as an example , gpt told me in encode-decoder attention which i am not implement it nor see it yet ( i even dont know whats it do )
        # Q comes from decoder while K and V comes from encoder
        # key=none -> key=value
        attention_output = self.attention_layer(query=inputs, value=inputs)
        attention_output = self.dropout1(attention_output)
        # now each toekn aware of whole sentence how ? i dont know :)
        #
        # look at the figur 1 in attentoins is all you need paper , you will find this layer there
        out1 = self.layernorm1(inputs + attention_output)
        fnn_output = self.feedforward_network(out1)
        fnn_output = self.dropout2(fnn_output)
        return self.layernorm2(out1 + fnn_output)
