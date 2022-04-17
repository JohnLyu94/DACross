import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dropout, MultiHeadAttention, Dense
from tensorflow.keras.layers import ReLU, ELU, PReLU
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPool1D


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = Dense(d_model)

    def _split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask):
        # (..., seq_len_q, seq_len_k)
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # scaling
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # masking
        if mask is not None:
            mask = tf.cast(tf.equal(mask, 0), tf.float32)
            mask = mask[:, tf.newaxis, tf.newaxis, :]
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(
            scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # (batch_size, num_heads, seq_len_q, depth)
        q = self._split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self._split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self._split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, num_heads=1, dropout=0.1):
        super(EncoderLayer, self).__init__()
        assert num_heads >= 1

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            # (batch_size, seq_len, dff)
            Dense(dff, activation='relu'),
            Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

    def call(self, x, mask, training=None):
        # (batch_size, input_seq_len, d_model)
        x_norm1 = self.layernorm1(x)
        attn_output, _ = self.mha(x_norm1, x_norm1, x_norm1, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out1 = x + attn_output

        x_norm2 = self.layernorm2(out1)
        ffn_output = self.ffn(x_norm2)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out2 = out1 + ffn_output

        return out2


class MultiHeadAffinityMap(tf.keras.layers.Layer):
    def __init__(self,
                 input_dims,
                 multi_head_num=1):
        super(MultiHeadAffinityMap, self).__init__()

        assert multi_head_num >= 1

        self.input_dims = input_dims
        self.multi_head_num = multi_head_num

        # initialize weights
        self.w_multi_head_affinity = []
        for k in range(self.multi_head_num):
            w_affinity = self.add_weight(
                shape=(self.input_dims, self.input_dims),
                initializer='glorot_uniform',
                name='affinity_weight_%d' % (k), trainable=True
            )

            self.w_multi_head_affinity.append(w_affinity)

    def call(self, inputs):
        """
            primary, secondary: (BATCH_SIZE, MAX_LEN_1|MAX_LEN_2, INPUT_DIMS)
            
            return:
                affinity_matrix: (BATCH_SIZE, HEAD_NUM, MAX_LEN_1, MAX_LEN_2)
        """

        primary, secondary = inputs

        # compute affinity feature map
        affinity_maps = tf.TensorArray(
            tf.float32, size=self.multi_head_num, dynamic_size=False, clear_after_read=False
        )

        for k in range(self.multi_head_num):
            w_affinity = self.w_multi_head_affinity[k]

            affinity_map = tf.keras.activations.tanh(
                tf.matmul(
                    tf.matmul(primary, w_affinity), secondary, transpose_b=True
                )
            )

            affinity_maps = affinity_maps.write(k, affinity_map)

        # (BATCH_SIZE, HEAD_NUM, ...,...)
        outputs = tf.transpose(affinity_maps.stack(), perm=[1, 0, 2, 3])
        affinity_maps = affinity_maps.close()
        return outputs


class CoAttentionFusion(tf.keras.layers.Layer):
    def __init__(self,
                 model_dims,
                 inner_dims,
                 multi_head_num=1,
                 dropout=0.,
                 activation='relu',
                 use_bias=True,
                 pooling='max'):
        super(CoAttentionFusion, self).__init__()

        assert multi_head_num >= 1
        assert inner_dims % multi_head_num == 0
        assert activation in ['relu', 'prelu', 'elu']
        assert pooling in ['max', 'mean']

        self.head_dims = inner_dims // multi_head_num
        self.model_dims = model_dims
        self.inner_dims = inner_dims
        self.multi_head_num = multi_head_num
        self.affinity_activation = activation
        self.activation = activation
        self.use_bias = use_bias

        if activation == 'relu':
            self.activation = ReLU()
        elif activation == 'elu':
            self.activation = ELU()
        elif activation == 'prelu':
            self.activation = PReLU()

        # initialize sub layers
        self.dropout_primary = Dropout(dropout)
        self.dropout_secondary = Dropout(dropout)

        self.affinity_layer = MultiHeadAffinityMap(
            self.model_dims, multi_head_num=self.multi_head_num)

        self.linear_primary = Dense(
            self.inner_dims, use_bias=False)
        self.linear_secondary = Dense(
            self.inner_dims, use_bias=False)

        if pooling == 'max':
            self.global_pooling_layer = GlobalMaxPool1D()
        elif pooling == 'mean':
            self.global_pooling_layer = GlobalAveragePooling1D()

        self.ffn_primary = Dense(
            self.model_dims, activation=activation, use_bias=self.use_bias)
        self.ffn_secondary = Dense(
            self.model_dims, activation=activation, use_bias=self.use_bias)

    def _split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.multi_head_num, self.head_dims))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, masks, training=None):
        """
            primary,secondary: (BATCH_SIZE, MAX_LEN_1|MAX_LEN_2, DIMS)
        """
        primary, secondary = inputs
        primary_mask, secondary_mask = masks
        batch_size = tf.shape(primary)[0]

        affinity_mask = tf.matmul(tf.expand_dims(primary_mask, 1), tf.expand_dims(secondary_mask, 1), transpose_a=True)
        affinity_mask = tf.cast(affinity_mask[:, tf.newaxis, :, :], tf.float32)
        # dropout
        primary = self.dropout_primary(primary, training=training)
        secondary = self.dropout_secondary(secondary, training=training)

        # affinity maps
        # (BATCH_SIZE, HEAD_NUM, MAX_LEN_1, MAX_LEN_2)
        affinity_maps = self.affinity_layer([primary, secondary])

        # masking
        affinity_maps = tf.multiply(affinity_maps, affinity_mask)

        projected_primary = self.linear_primary(primary)
        projected_secondary = self.linear_secondary(secondary)

        # split into K heads
        projected_primary = self._split_heads(projected_primary, batch_size)
        projected_secondary = self._split_heads(projected_secondary, batch_size)

        # matmul by affinity maps
        weighted_primary = self.activation(tf.matmul(tf.transpose(
            affinity_maps, perm=[0, 1, 3, 2]), projected_primary))
        weighted_secondary = self.activation(
            tf.matmul(affinity_maps, projected_secondary))

        # pooling
        weighted_primary = tf.unstack(weighted_primary, batch_size, axis=0)
        weighted_secondary = tf.unstack(weighted_secondary, batch_size, axis=0)

        for k in range(batch_size):
            if k == 0:
                pooling_primary = tf.expand_dims(self.global_pooling_layer(
                    tf.transpose(weighted_primary[k], perm=[1, 0, 2])), 0)
                pooling_secondary = tf.expand_dims(self.global_pooling_layer(
                    tf.transpose(weighted_secondary[k], perm=[1, 0, 2])), 0)
            else:
                pooling_primary = tf.concat([pooling_primary, tf.expand_dims(self.global_pooling_layer(
                    tf.transpose(weighted_primary[k], perm=[1, 0, 2])), 0)], 0)
                pooling_secondary = tf.concat([pooling_secondary, tf.expand_dims(self.global_pooling_layer(
                    tf.transpose(weighted_secondary[k], perm=[1, 0, 2])), 0)], 0)

        output_primary = tf.concat([primary, pooling_secondary], -1)
        output_secondary = tf.concat([secondary, pooling_primary], -1)

        output_primary = self.ffn_primary(output_primary)
        output_secondary = self.ffn_secondary(output_secondary)

        return output_primary, output_secondary


if __name__ == '__main__':
    import time

    # create a example of inputs, (3, 5, 32)
    ex_1 = tf.random.uniform((4, 32))
    ex_2 = tf.random.uniform((3, 32))
    ex_3 = tf.random.uniform((5, 32))

    ex_1 = tf.expand_dims(tf.pad(ex_1, tf.constant([[0, 1], [0, 0]])), 0)
    ex_2 = tf.expand_dims(tf.pad(ex_2, tf.constant([[0, 2], [0, 0]])), 0)
    ex_3 = tf.expand_dims(tf.pad(ex_3, tf.constant([[0, 0], [0, 0]])), 0)

    primary = tf.concat([ex_1, ex_2, ex_3], 0)

    ex_1 = tf.random.uniform((5, 32))
    ex_2 = tf.random.uniform((4, 32))
    ex_3 = tf.random.uniform((6, 32))

    ex_1 = tf.expand_dims(tf.pad(ex_1, tf.constant([[0, 1], [0, 0]])), 0)
    ex_2 = tf.expand_dims(tf.pad(ex_2, tf.constant([[0, 2], [0, 0]])), 0)
    ex_3 = tf.expand_dims(tf.pad(ex_3, tf.constant([[0, 0], [0, 0]])), 0)

    secondary = tf.concat([ex_1, ex_2, ex_3], 0)

    co_attentive_layer = MultiHeadAffinityMap(32, multi_head_num=4)
    co_attention_fusion = CoAttentionFusion(32, 32 * 4, multi_head_num=4)
    transformer_encoder = EncoderLayer(32, 32 * 4, num_heads=4)

    mask = tf.constant([
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1]
    ])

    st = time.time()
    outputs = co_attentive_layer([primary, secondary])
    ed_1 = time.time()
    print(ed_1 - st, 's')
    outputs_2 = co_attention_fusion([primary, secondary])
    ed_2 = time.time()
    print(ed_2 - ed_1, 's')
    outputs_3 = transformer_encoder(primary, mask)
    ed_3 = time.time()
    print(ed_3 - ed_2, 's')
