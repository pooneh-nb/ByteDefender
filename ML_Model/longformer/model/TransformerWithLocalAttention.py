import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from BlockBasedLocalAttention import BlockBasedLocalAttention

class TransformerWithLocalAttention(layers.Layer):
    def __init__(self, embed_dim, block_size, left_context, right_context, ff_dim, rate=0.1):
        super(TransformerWithLocalAttention, self).__init__()
        # replace the custom LocalWindowedAttention with the multihead attention in regular transformer
        self.att = BlockBasedLocalAttention(embed_dim=embed_dim, block_size=block_size, 
                                            left_context= left_context, right_context=right_context)
        
        self.ffn = Sequential([
            layers.Dense(ff_dim, activation="relu"), 
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
