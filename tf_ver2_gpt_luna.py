import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding, LayerNormalization)

# Multi-Head Attention Layer. #
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(
        self, d_model, n_heads, 
        name="multi_head_attn_layer"):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        d_depth = int(d_model / n_heads)
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_depth = d_depth
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.wc = tf.keras.layers.Dense(d_model)
    
    def split_heads(self, x):
        # Input is (batch_size, seq_len, d_model). #
        # Output is (batch_size, num_heads, seq_len, depth). #
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        output_shp = (batch_size, seq_length, 
                      self.n_heads, self.d_depth)
        
        x = tf.reshape(x, output_shp)
        return tf.transpose(x, [0, 2, 1, 3])
    
    def combine_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[2]
        output_shp = (
            batch_size, seq_length, self.d_model)
        
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, output_shp)
    
    def matmul_prefix_sum(self, x, y, z):
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[2]
        one_vector = tf.ones(
            [batch_size, 1], 
            dtype=tf.float32, name="one_t")
        
        seq_index = tf.cast(tf.expand_dims(
            tf.range(seq_length), axis=0), tf.float32)
        seq_index = tf.matmul(
            one_vector, seq_index + 1.0)
        seq_index = tf.expand_dims(
            tf.expand_dims(seq_index, axis=1), axis=3)
        
        x = tf.divide(x, seq_index)
        x = tf.expand_dims(x, axis=3)
        y = tf.expand_dims(y, axis=4)
        z = tf.expand_dims(z, axis=3)
        yz = tf.matmul(y, z)
        
        yz_prefix  = tf.math.cumsum(yz, axis=2)
        xyz_causal = tf.squeeze(
            tf.matmul(x, yz_prefix), axis=3)
        return xyz_causal
    
    def call(self, v, k, q, p):
        norm_d = tf.math.rsqrt(float(self.d_depth))
        
        p = self.split_heads(p)
        q = self.split_heads(self.wq(q) * norm_d)
        k = self.split_heads(self.wk(k))
        v = self.split_heads(self.wv(v))
        
        attn_pack = tf.add(tf.nn.elu(
            tf.matmul(p, q, transpose_b=True)), 1.0)
        attn_pack = tf.transpose(attn_pack, [0, 1, 3, 2])
        
        attn_unpack = self.matmul_prefix_sum(
            q, k, attn_pack)
        attn_unpack = tf.nn.softmax(attn_unpack)
        attn_outputs = self.matmul_prefix_sum(
            attn_unpack, attn_pack, v)
        attn_outputs = self.wc(
            self.combine_heads(attn_outputs))
        return attn_outputs
        
class FFWNetwork(tf.keras.layers.Layer):
    def __init__(self, d_ffwd, d_model):
        super(FFWNetwork, self).__init__()
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        
        self.ffwd_1 = tf.keras.layers.Dense(
            d_ffwd, activation="relu")
        self.ffwd_2 = tf.keras.layers.Dense(d_model)
    
    def call(self, x):
        return self.ffwd_2(self.ffwd_1(x))

# GPT Decoder Layer. #
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self, d_model, n_heads, d_ffwd, 
        rate1=0.1, rate2=0.1, name="decoder_layer"):
        super(DecoderLayer, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.ffwd_self = FFWNetwork(d_ffwd, d_model)
        self.attn_self = MultiHeadAttention(
            d_model, n_heads, name=name)
        
        self.lnorm_1 = LayerNormalization(epsilon=1e-6)
        self.lnorm_2 = LayerNormalization(epsilon=1e-6)
        self.dropout_1 = tf.keras.layers.Dropout(rate1)
        self.dropout_2 = tf.keras.layers.Dropout(rate2)
    
    def call(
        self, x_enc, x_p, x_pos, training=True):
        x_embed = x_enc + x_pos
        
        # Decoder only Attention Mechanism. #
        attn_self_output = self.attn_self(
            x_embed, x_embed, x_embed, x_p)
        attn_self_output = self.dropout_1(
            attn_self_output, training=training)
        attn_self_output = tf.add(
            x_embed, self.lnorm_1(attn_self_output))
        
        ffwd_self_output = self.lnorm_2(
            self.ffwd_self(attn_self_output))
        ffwd_self_output = tf.add(
            attn_self_output, ffwd_self_output)
        ffwd_self_output = self.dropout_2(
            ffwd_self_output, training=training)
        return ffwd_self_output

class Decoder(tf.keras.layers.Layer):
    def __init__(
        self, n_layers, d_model, 
        n_heads, d_ffwd, vocab_size, 
        max_seq_length, p_len, rate1=0.1, rate2=0.1):
        super(Decoder, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.p_len = p_len
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.d_rsqrt = tf.math.sqrt(
            tf.cast(d_model, tf.float32))
        self.vocab_size = vocab_size
        
        # Embedding layers. #
        tmp_pos_embed = []
        for n_layer in range(n_layers):
            tmp_pos_embed.append(
                Embedding(max_seq_length, d_model))
        
        self.pos_embed = tmp_pos_embed
        self.dec_embed = Embedding(vocab_size, d_model)
        del tmp_pos_embed
        
        tmp_p_luna = []
        for n_layer in range(n_layers):
            tmp_p_luna.append(
                Embedding(p_len, d_model))
        self.p_luna = tmp_p_luna
        del tmp_p_luna
        
        # Decoder Layers. #
        tmp_dec_layers = []
        for n_layer in range(n_layers):
            tmp_dec_layers.append(DecoderLayer(
                d_model, n_heads, d_ffwd, 
                rate1=rate1, rate2=rate2, 
                name="decoder_layer_" + str(n_layer+1)))
        
        self.dec_layers = tmp_dec_layers
        self.emb_dropout = tf.keras.layers.Dropout(rate1)
        del tmp_dec_layers
    
    def call(self, x, training=True):
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        one_vector = tf.ones(
            [batch_size, 1], 
            dtype=tf.int32, name="one_vector")
        
        p_index = tf.expand_dims(
            tf.range(self.p_len), axis=0)
        p_index = tf.matmul(
            one_vector, p_index)
        
        x_pos_index = tf.expand_dims(
            tf.range(seq_length), axis=0)
        x_tok_embed = tf.multiply(
            self.d_rsqrt, self.dec_embed(x))
        x_tok_embed = self.emb_dropout(
            x_tok_embed, training=training)
        
        layer_input = x_tok_embed
        for m in range(self.n_layers):
            x_p_input = self.p_luna[m](p_index)
            
            x_pos_embed = tf.multiply(
                self.d_rsqrt, 
                self.pos_embed[m](x_pos_index))
            x_pos_embed = self.emb_dropout(
                x_pos_embed, training=training)
            
            layer_output = self.dec_layers[m](
                layer_input, x_p_input, 
                x_pos_embed, training=training)
            layer_input  = layer_output
        return layer_output

class GPTLuna(tf.keras.Model):
    def __init__(
        self, n_layers, n_heads, 
        d_model, d_ffwd, vocab_size, 
        max_seq_length, p_len, rate1=0.1, rate2=0.1):
        super(GPTLuna, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.p_len = p_len
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.vocab_size = vocab_size
        
        # Output projection. #
        self.gpt_model = Decoder(
            n_layers, d_model, n_heads, 
            d_ffwd, vocab_size, max_seq_length, 
            p_len, rate1=rate1, rate2=rate2)
        self.p_decoder = tf.keras.layers.Dense(vocab_size)
    
    def call(self, x, training=True):
        dec_outputs = self.gpt_model(
            x, training=training)
        dec_logits  = self.p_decoder(dec_outputs)
        return dec_logits
    
    def infer(self, x):
        input_len = tf.shape(x)[1]
        infer_ids = [tf.expand_dims(x[:, 0], axis=1)]
        
        for step in range(self.seq_len):
            tmp_inputs = tf.concat(infer_ids, axis=1)
            tmp_logits = self.call(tmp_inputs, training=False)
            
            tmp_logit = tmp_logits[:, -1, :]
            tmp_index = tf.cond(
                step < (input_len-1), 
                lambda: x[:, step+1], 
                lambda: tf.argmax(
                    tmp_logit, axis=-1, output_type=tf.int32))
            infer_ids.append(tf.expand_dims(tmp_index, axis=1))
        return tf.concat(infer_ids, axis=1)
        

