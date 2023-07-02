import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class model(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, maxlen, vocab_size,num_class, rate = 0.1):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim = vocab_size, output_dim= embed_dim)
        self.pos_emb = layers.Embedding(input_dim = maxlen, output_dim = embed_dim)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation = 'relu'),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon = 1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon = 1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.avgpooling = layers.GlobalAveragePooling1D()
        self.output_layer = layers.Dense(num_class, activation = 'softmax')

    
    def call(self, x, training):
        maxlen = tf.shape(x)[0]
        positions = tf.range(start = 0, limit = maxlen, delta = 1)  
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        x = x+positions
        attn_output = self.attn(x, x)
        attn_output = self.dropout1(attn_output, training = training)
        out1 = self.layernorm1(x+attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training = training)
        ffn_output = self.layernorm2(out1 + ffn_output)
        ffn_output = self.avgpooling(ffn_output)
        ffn_output = self.layernorm3(ffn_output , training = training)
        output = self.output_layer(ffn_output)
        return output