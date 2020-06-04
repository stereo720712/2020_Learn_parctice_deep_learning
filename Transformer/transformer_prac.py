'''
Sean ,practic nn
transformer prac
encoder:
fnn:
self-attention:

decoder:
fnn:
encode-decode-attention:
self-attention:
'''
# ====================================================
'''
self-attention:
define: give a Q, decide which Values by Q and K relation
input-vector X generate:
query,key,value vector
(define three weight: Wq,Wk,Wv; XWq:Q,XWk:K,XWv:V )

ex: X1:

1. score:    Q1xK1           Q1xK2                        ... n words
2. /8        Q1xK1/8         Q1xK2/8                     (square root of Key dimension )
3.softmax     0.37            0.25
4.          V1 x softmax   v2 x softmax
5.sum the vector in 4 => Z
'''

# =================================================================
'''
Multi-headed attention. 
purpose: give the self-attention layer multi representation subspace
define:
expend to multi Q/K/V weight set,(Google Transformer use 8)
every encoder/decoder have 8 matrix set
evert set used to project the "word embedding" to different subspace

Assemble:
The transformer establish by multi Attention layer with different weight(parallel in one level)
ex:
Will have 8 Weight set [(Wq1,Wk1,WV1),(Wq2,Wk2,WV2)...]
and get 8 sum attention head Z0, Z1, Z2,....

BUT, FNN only need one matrix , we need zip it, and multiply  some attention weight matrix

[Z0,Z1....Z8] x W0 ==> get a union all attention info. matrix Z

'''
# =========================================================
'''
A problem when  two word is same : ex : I saw a saw saw a saw.
the second saw and the forth saw will get the same out because of the same input
so we need a encoding for  time sequence relation(order) about word order in sentence

We add a vector for the word location in sentence.
try to see Position encoding in transformer  in PTT(week 17)
# https://tobiaslee.top/2018/12/13/Start-from-Transformer/

'''
import tensorflow as tf
import numpy as np
import  matplotlib.pyplot as plt

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'


def positional_encoding(pos,d_model):

    def get_angles(position,i):
        # i相當於公式裡面的2i or 2i+1
        # 返回 shape=[position_num, d_model]
        return position / np.power(10000.,2.*(i // 2.) /np.float(d_model))

    angle_rates = get_angles(np.arange(pos)[:, np.newaxis],
                             np.arange(d_model)[np.newaxis, :])

    # 2i position use sin function, 2i+1 use cos function
    pe_sin = np.sin(angle_rates[:, 0::2])
    pe_cos = np.cos(angle_rates[:, 1::2])
    pos_encoding = np.concatenate([pe_sin, pe_cos], axis=-1)
    pos_encoding = tf.cast(pos_encoding[np.newaxis, ...], tf.float32) # change to tf type && add a axis for batch
    return  pos_encoding


def scaled_dot_product_attention(q, k, v, mask):

    matmul_qk = tf.matmul(q, k, transpose_b=True)

    #使用dk進行縮放????
    dk = tf.cast(tf.shape(q)[-1], tf.float32)
    scaled_attention = matmul_qk / tf.math.sqrt(dk)
    #掩碼 mask
    if mask is not None:
        # 將mask的 token 乘 -1e-9 這樣與attention相加後,mask的位置經過softmax後就為0 ???????
        # padding位置 mask = 1
        scaled_attention += mask * -1e-9
    # 通過softmax獲取attention權重 ,mask部分 softmax後為0
    attention_weights = tf.nn.softmax(scaled_attention) # shape=[batch_size,seq_lne_q,seq_len_k]
    # 乘以value
    outputs = tf.matmul(attention_weights, v) # shape=[batch_size,seq_len_q,seq_len_k]
    return outputs, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_models = d_model
        # d_model 必須可以正確分成多個頭
        assert d_model % num_heads == 0
        # 分頭之後維度
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        #分頭 將頭個數的維度, 放到seq_len前面, x輸入shape=[batch_size,seq_len, d_model]
        x = tf.reshape(x,[batch_size, -1, self.num_heads,self.depth]) # https://blog.csdn.net/lxg0807/article/details/53021859
  
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        # 分頭前的網路 根據q,k,v的輸入 計算 Q,K,V的語義
        q = self.wq(q)  # shape=[batch_size, seq_len, d_model]
        k = self.wq(k)
        v = self.wq(v)

        #分頭
        q = self.split_heads(q, batch_size)  # shape=[batch_size, num_heads, seq_len, depth]
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # 把多頭維度後移
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) #shape=[batch_size,seq_len_q,num_heads,depth]
  
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_models)) #shape=[batch_size,seq_len_q,d_model]

        output = self.dense(concat_attention)
        return output, attention_weights

class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        super(LayerNormalization,self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self,input_shape):
        self.gamma = self.add_weight(name="gamma",
                                     shape=input_shape[-1:],
                                     initializer=tf.ones_initializer(),
                                     trainable=True)
        self.beta = self.add_weight(name="beta",
                                    shape=input_shape[-1:],
                                    initializer=tf.ones_initializer(),
                                    trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):  # x shape = [batch_size, seq_len, d_model]
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta


#point_wise 前向網路
def point_wise_feed_forward(d_model, diff):
    return tf.keras.Sequential(
        [tf.keras.layers.Dense(diff, activation=tf.nn.relu),
         tf.keras.layers.Dense(d_model)])


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward(d_model, dff)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.droupout1 = tf.keras.layers.Dropout(dropout_rate)
        self.droupout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self,inputs, training, mask):
        # multi head attention (Q = K = V When Encoding)
        att_output, _ = self.mha(inputs, inputs, inputs , mask)
        att_output = self.droupout1(att_output, training=training)
        output1 = self.layernorm1(inputs + att_output) # 殘差 , shaoe=[batch_size, seq_len, d_model]

        # feed forward network
        ffn_output = self.ffn(output1)
        ffn_output = self.droupout2(ffn_output,training=training)
        output2 = self.layernorm2(output1 + ffn_output) # shape=[batch_size, seq_len, d_model]
        return output2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_layers, num_heads, dff,
                 input_vocab_size, max_seq_len, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.emb = tf.keras.layers.Embedding(input_vocab_size, d_model)  # shape=[batch_size, seq_len, d_model]
        self.pos_encoding = positional_encoding(max_seq_len, d_model)  # shape=[1,max_seq_len,d_model]
        self.encoder_layer = [EncoderLayer(d_model, num_heads, dff, dropout_rate)
                              for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training, mask):
        # 輸入部分:input shape = [batch_size,seq_len]
        seq_len = inputs.shape[1]  #句子真实长度
        word_embedding = self.emb(inputs) # shape = [batch_size, seq_len, d_model]
        word_embedding *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        emb = word_embedding + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(emb, training=training)
        for i in range(self.num_layers):
            x = self.encoder_layer[i](x, training, mask)
        return x


#padding mask
def create_padding_mask(seq):
    '''為了避免輸入padding的token 對句子語意的影響 , 需要將padding位 mask掉,
       原來為0的pdding項的mask輸出為1 , encoder and decoder过程都会用到
    '''
    seq = tf.cast(tf.math.equal(seq,0),tf.float32)
    # 擴充維度以便於使用attentionq matrix , sqq輸入 shape=[batch_size, seq_len] 輸出：shape=[batch_size, 1, 1, seq_len]
    return seq[:, np.newaxis, np.newaxis, :]


#look-ahead-mask
def create_look_ahead_mask(size):

    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # shape=[seq_len,seq_len]

def create_mask(inputs, targets):

    # encoder  only have padding_mask
    encoder_padding_mask = create_padding_mask(inputs)

    #decoder padding mask 用于第二层的 multi-head attention
    decoder_padding_mask = create_padding_mask(inputs)

    #seq_mask, mask掉未預測的詞
    seq_mask = create_look_ahead_mask(tf.shape(targets)[1])

    #decoder_targets_padding_mask 解碼層的輸入mask
    decoder_targets_padding_mask = create_padding_mask(targets)

    # 合併解碼層mask, 用於 第一層 masked multi-head-attention
    look_ahead_task = tf.maximum(decoder_targets_padding_mask, seq_mask)
    return encoder_padding_mask, look_ahead_task, decoder_padding_mask


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward(d_model, dff)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, encoder_out, training, look_ahead_mask, padding_mask):
        # masked multi-head attention : Q = K = V
        att_out1, att_weight1 = self.mha1(inputs, inputs, inputs, look_ahead_mask)
        att_out1 = self.dropout1(att_out1, training=training)
        att_out1 = self.layernorm1(inputs + att_out1)

        # multi-head attention: Q = att_out1, K = V = encoder_out
        att_out2, att_weight2 = self.mha2(att_out1, encoder_out, encoder_out, padding_mask)
        att_out2 = self.dropout2(att_out2, training=training)
        att_out2 = self.layernorm2(att_out1 + att_out2)

        #feed forward work
        ffn_out = self.ffn(att_out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        output = self.layernorm3(att_out2 + ffn_out)
        return output, att_weight1, att_weight2


class Decoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_layers, num_heads, dff,
                 target_vocab_size, max_seq_len, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.seq_len = tf.shape  # ?????
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.word_embedding = tf.keras.layers.Embedding(target_vocab_size,d_model)
        self.pos_encoding = positional_encoding(max_seq_len, d_model)
        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate)
                               for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)


    def call(self, inputs, encoder_out, training, look_ahead_mask, padding_mask):
        seq_len = inputs.shape[1]
        attention_weights = {}
        word_embedding = self.word_embedding(inputs)
        word_embedding *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        emb = word_embedding + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(emb, training=training)

        for i in range(self.num_layers):
            x, att1, att2 = self.decoder_layers[i](x, encoder_out, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_att_w1'.format(i+1)] = att1
            attention_weights['decoder_layer{}_att_w2'.format(i+1)] = att2

        return x, attention_weights


# include encoder , decoder , and ffn ,解碼層的輸出經過線性層後得到transformer的輸出
class Transformer(tf.keras.Model):
    def __init__(self, d_model, num_layers, num_heads, dff,
                 input_vocab_size, target_vocab_size, max_seq_len, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, num_layers, num_heads, dff, input_vocab_size, max_seq_len, dropout_rate)
        self.decoder = Decoder(d_model, num_layers, num_heads, dff, target_vocab_size, max_seq_len, dropout_rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self,inputs, targets, training, encoder_padding_mask,
             look_ahead_mask, decoder_padding_mask):
        # encoder progress, output shape = [batch_size, seq_len_input, d_model]
        encoder_output = self.encoder(inputs, training, encoder_padding_mask)
        # decoder progress , output shape = [batch_size, seq_len_target, d_model]
        decoder_output, att_weights = self.decoder(targets, encoder_output, training,
                                                   look_ahead_mask,decoder_padding_mask)

        # final mapping to output layer
        final_out = self.final_layer(decoder_output) #shape = [batch_size, seq_len_target, target_vocab_size]
        return final_out, att_weights



#test

sample_trasformer = Transformer(
    num_layers=2, d_model=512, num_heads=8, dff=1024,
    input_vocab_size=8500, target_vocab_size=8000, max_seq_len=120
)
temp_input = tf.random.uniform((64,62))
temp_target = tf.random.uniform((64,26))

fn_out, att = sample_trasformer(temp_input, temp_target, training=False,
                                encoder_padding_mask=None, look_ahead_mask=None,
                                decoder_padding_mask=None)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
output = sess.run([fn_out])
print(output)
print(fn_out)
print(att['decoder_layer1_att_w1'].shape)
print(att['decoder_layer1_att_w2'].shape)






