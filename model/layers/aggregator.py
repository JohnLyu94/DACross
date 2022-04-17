import random
import tensorflow as tf

from tensorflow.keras.layers import ReLU, PReLU, ELU, Dropout, LeakyReLU, Dense


def random_sum_to(n, num_terms=None):
    num_terms = (num_terms or random.randint(2, n)) - 1
    a = random.sample(range(1, n), num_terms) + [0, n]
    list.sort(a)
    return [a[i+1] - a[i] for i in range(len(a) - 1)]


def node_segment(n_src, n_neighbors):
    neighbor_node_idxs = tf.repeat(tf.range(n_src), n_neighbors // n_src)
    src_node_idxs = tf.range(n_src)
    return neighbor_node_idxs, tf.concat([neighbor_node_idxs, src_node_idxs], axis=0), n_src


class MeanAggregator(tf.keras.layers.Layer):
    def __init__(self,
                 output_dims,
                 use_bias=False,
                 dropout=0.,
                 activation='relu',
                 fusion_method='concat'):
        super(MeanAggregator, self).__init__()

        assert fusion_method in ['concat', 'sum']
        if fusion_method == 'concat':
            self.aggregate_dims = output_dims // 2
        else:
            self.aggregate_dims = output_dims

        assert activation in ['relu', 'prelu']
        if activation == 'prelu':
            self.activation = PReLU()
        else:
            self.activation = ReLU()

        self.dropout_layer = Dropout(dropout)
        self.fusion_method = fusion_method

        # weight and bias
        self.linear_proj_src = Dense(self.aggregate_dims, use_bias=use_bias)
        self.linear_proj_neighbor = Dense(self.aggregate_dims, use_bias=use_bias)

    def call(self, inputs, training=None):
        """

        Args:
            inputs: [n_nodes, n_features], [n_nodes*n_sample, n_features]
            training: True or False

        Returns: [n_nodes, hidden_dims]

        """
        src_vectors, neighbor_vectors = inputs

        neighbor_vectors = self.dropout_layer(
            neighbor_vectors, training=training)

        # element-wise mean
        # acquired by src_node : neighbor_nodes
        # segment indexing
        neighbor_node_idxs, _, n_segments = node_segment(
            src_vectors.shape[0], neighbor_vectors.shape[0])
        # [n_nodes, n_features]
        neighbor_means = tf.math.unsorted_segment_mean(neighbor_vectors,
                                                       neighbor_node_idxs,
                                                       num_segments=n_segments)

        # w[dot]h
        neighbor_hidden_vectors = self.linear_proj_neighbor(neighbor_means)
        self_hidden_vectors = self.linear_proj_src(src_vectors)

        if self.fusion_method == 'concat':
            aggregated_vectors = tf.concat(
                [self_hidden_vectors, neighbor_hidden_vectors], -1)
        elif self.fusion_method == 'sum':
            aggregated_vectors = tf.add_n(
                [self_hidden_vectors, neighbor_hidden_vectors])

        return self.activation(aggregated_vectors)


class GCNAggregator(tf.keras.layers.Layer):
    def __init__(self,
                 output_dims,
                 dropout=0.,
                 use_bias=False,
                 activation='relu'):
        super(GCNAggregator, self).__init__()

        assert activation in ['relu', 'prelu']

        self.dropout_layer = Dropout(dropout)
        if activation == 'relu':
            activation = ReLU()
        else:
            activation = PReLU()
        self.ffn_layer = Dense(
            output_dims, activation=activation, use_bias=use_bias)

    def call(self, inputs, training=None):
        src_vectors, neighbor_vectors = inputs

        neighbor_vectors = self.dropout_layer(
            neighbor_vectors, training=training)

        # element-wise mean {h_src} U {h_neighbors}
        _, neighbor_src_node_idxs, n_segments = node_segment(src_vectors.shape[0],
                                                             neighbor_vectors.shape[0])
        # concat -> [n_neighbors+n_src, n_features]
        means = tf.math.unsorted_segment_mean(tf.concat([neighbor_vectors, src_vectors], axis=0),
                                              neighbor_src_node_idxs,
                                              num_segments=n_segments)

        # w[dot]h
        aggregated_vectors = self.ffn_layer(means)

        return aggregated_vectors


class MaxPoolingAggregator(MeanAggregator):
    def __init__(self,
                 output_dims,
                 dropout=0.,
                 use_bias=False,
                 activation='relu',
                 fusion_method='concat'):
        super(MaxPoolingAggregator, self).__init__(output_dims,
                                                   use_bias=use_bias,
                                                   activation=activation,
                                                   dropout=dropout,
                                                   fusion_method=fusion_method)

        # pooling_aggregator = MAX(ACT(W_pool*h_neighbor+b_pool))
        # neighbor_mlp_w & neighbor_mlp_b
        self.ffn_pool_layer = Dense(
            self.aggregate_dims * 4, activation='relu', use_bias=use_bias)

    def call(self, inputs, training=None):
        src_vectors, neighbor_vectors = inputs

        neighbor_vectors = self.dropout_layer(
            neighbor_vectors, training=training)

        # Dense layer (MLP) before element-wise max operation
        neighbor_vectors = self.ffn_pool_layer(neighbor_vectors)

        neighbor_node_idxs, _, n_segments = node_segment(
            src_vectors.shape[0], neighbor_vectors.shape[0])

        neighbor_max = tf.math.unsorted_segment_max(neighbor_vectors,
                                                    neighbor_node_idxs,
                                                    num_segments=n_segments)

        # w[dot]h
        neighbor_hidden_vectors = self.linear_proj_neighbor(neighbor_max)
        self_hidden_vectors = self.linear_proj_src(src_vectors)

        if self.fusion_method == 'concat':
            aggregated_vectors = tf.concat(
                [self_hidden_vectors, neighbor_hidden_vectors], axis=1)
        elif self.fusion_method == 'sum':
            aggregated_vectors = tf.add_n(
                [self_hidden_vectors, neighbor_hidden_vectors])

        return self.activation(aggregated_vectors)


class AttentionAggregator(tf.keras.layers.Layer):
    def __init__(self,
                 weight_dims,
                 num_heads,
                 dropout=0.,
                 use_bias=False,
                 activation='relu',
                 fusion_method='concat'):
        super(AttentionAggregator, self).__init__()
        assert fusion_method in ['concat', 'avg']
        assert num_heads >= 1

        self.weight_dims = weight_dims

        self.fusion_method = fusion_method

        self.num_heads = num_heads

        self.dropout_layer = Dropout(dropout)
        # set as the same as mentioned in origin
        assert activation in ['relu', 'prelu', 'elu']
        if activation == 'relu':
            self.activation = ReLU()
        elif activation == 'prelu':
            self.activation = PReLU()
        elif activation == 'elu':
            self.activation = ELU()

        self.ffn_activation = LeakyReLU(alpha=0.2)

        # init weights
        self.linear_layer = Dense(
            self.weight_dims *
            self.num_heads,
            use_bias=False)
        self.attention_layer = Dense(
            self.num_heads,
            use_bias=use_bias)

    def _split_heads(self, x, batch_size, dims):

        x = tf.reshape(x, (batch_size, -1, self.num_heads, dims))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None, training=None):

        node_batch = inputs  # (BATCH_SIZE, MAX_NODE, INPUT_DIMS)
        batch_size = tf.shape(node_batch)[0]
        max_neighbor = tf.shape(node_batch)[1]

        node_batch = self.dropout_layer(node_batch, training=training)

        # linear transformation
        linear_proj = self.linear_layer(node_batch)

        for i in range(batch_size):
            if i == 0:
                src_nodes = tf.expand_dims(
                    tf.gather_nd(linear_proj, [[i, 0]]), 0)
            else:
                src_nodes = tf.concat(
                    [src_nodes,
                     tf.expand_dims(tf.gather_nd(linear_proj, [[i, 0]]), 0)], 0)
        src_nodes = tf.tile(src_nodes, [1, max_neighbor, 1])

        concat_linear = tf.concat([src_nodes, linear_proj], -1)

        attention_node = self.attention_layer(concat_linear)

        attention_weights = self.ffn_activation(attention_node)

        attention_weights = tf.squeeze(self._split_heads(
            attention_weights, batch_size, 1))  # (batch_size, head_num, max_neighbor)
        linear_proj = self._split_heads(
            linear_proj, batch_size, self.weight_dims)

        if mask is not None:
            attention_weights += tf.broadcast_to(
                tf.expand_dims(
                    tf.cast(tf.equal(mask, 0),
                            tf.float32) * -1e9, axis=1),
                (batch_size, self.num_heads, max_neighbor)
            )
        # (batch_size, head_num, max_neighbor, 1)
        attention_weights = tf.expand_dims(
            tf.nn.softmax(attention_weights, axis=2), 3)

        weighted_hiddens = tf.math.multiply(linear_proj, attention_weights)
        outputs = tf.math.reduce_sum(
            weighted_hiddens, 2)  # sum src and neighbors

        if self.fusion_method == 'concat':
            return self.activation(tf.concat(tf.unstack(weighted_hiddens, axis=1), -1))
        elif self.fusion_method == 'avg':
            return self.activation(tf.math.reduce_mean(outputs, axis=1))


if __name__ == '__main__':
    # test
    import time
    import random

    # parameters
    BATCH_SIZE = 32
    INPUT_DIMS = 128
    OUTPUT_DIMS = 256
    NEIGHBOR_SIZE = 3
    src_nodes = tf.random.uniform((BATCH_SIZE, INPUT_DIMS))
    neighbor_nodes = tf.random.uniform(
        (BATCH_SIZE * NEIGHBOR_SIZE, INPUT_DIMS))
    att_nodes = tf.random.uniform((BATCH_SIZE, NEIGHBOR_SIZE+1, INPUT_DIMS))

    # test inputs for attention aggregator
    ex_1 = tf.random.uniform((4, INPUT_DIMS))
    ex_2 = tf.random.uniform((3, INPUT_DIMS))
    ex_3 = tf.random.uniform((5, INPUT_DIMS))

    ex_1 = tf.expand_dims(tf.pad(ex_1, tf.constant([[0, 1], [0, 0]])), 0)
    ex_2 = tf.expand_dims(tf.pad(ex_2, tf.constant([[0, 2], [0, 0]])), 0)
    ex_3 = tf.expand_dims(tf.pad(ex_3, tf.constant([[0, 0], [0, 0]])), 0)

    nodes = tf.concat([ex_1, ex_2, ex_3], 0)
    mask = tf.constant([
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1]
    ])
    mask = None
    # init aggregators
    mean_aggregator = MeanAggregator(OUTPUT_DIMS,
                                     use_bias=False,
                                     activation='prelu',
                                     dropout=0.5,
                                     fusion_method='concat')
    gcn_aggregator = GCNAggregator(OUTPUT_DIMS,
                                   use_bias=False,
                                   activation='prelu',
                                   dropout=0.5)
    max_pooling_aggregator = MaxPoolingAggregator(OUTPUT_DIMS,
                                                  use_bias=False,
                                                  activation='prelu',
                                                  dropout=0.5,
                                                  fusion_method='concat')
    attention_aggregator = AttentionAggregator(OUTPUT_DIMS,
                                               num_heads=4,
                                               dropout=0.2,
                                               activation='prelu',
                                               fusion_method='concat')

    st = time.time()
    r0 = mean_aggregator((src_nodes, neighbor_nodes), training=True)
    t0 = time.time()
    print(t0 - st, 's')
    r1 = gcn_aggregator((src_nodes, neighbor_nodes), training=True)
    t1 = time.time()
    print(t1 - t0, 's')
    r2 = max_pooling_aggregator((src_nodes, neighbor_nodes), training=True)
    t2 = time.time()
    print(t2 - t1, 's')
    r3 = attention_aggregator(att_nodes, mask=mask, training=True)
    t3 = time.time()
    print(t3 - t2, 's')
