import tensorflow as tf

from aggregator import MeanAggregator, GCNAggregator, MaxPoolingAggregator
from aggregator import random_sum_to


class GraphSAGELayer(tf.keras.layers.Layer):
    def __init__(self,
                 output_dims,
                 use_bias=False,
                 activation='relu',
                 dropout=0.,
                 fusion_method='concat',
                 aggregate_method='mean'):
        super(GraphSAGELayer, self).__init__()
        assert activation in ['relu', 'prelu']
        assert fusion_method in ['concat', 'sum', None]
        if fusion_method == 'concat':
            assert output_dims % 2 == 0
        assert aggregate_method in ['mean', 'gcn', 'max']

        # instances of aggregators
        if aggregate_method == 'mean':
            self.aggregator = MeanAggregator(output_dims,
                                             use_bias=use_bias,
                                             activation=activation,
                                             dropout=dropout,
                                             fusion_method=fusion_method)
        elif aggregate_method == 'gcn':
            self.aggregator = GCNAggregator(output_dims,
                                            use_bias=use_bias,
                                            activation=activation,
                                            dropout=dropout)
        elif aggregate_method == 'max':
            self.aggregator = MaxPoolingAggregator(output_dims,
                                                   use_bias=use_bias,
                                                   activation=activation,
                                                   dropout=dropout,
                                                   fusion_method=fusion_method)

    def call(self, inputs, training=None):
        hidden_vectors = self.aggregator(inputs, training=training)

        return hidden_vectors


if __name__ == '__main__':
    # test
    import time

    # parameters
    BATCH_SIZE = 32
    INPUT_DIMS = 128
    OUTPUT_DIMS = 256
    NEIGHBOR_SIZE = 3
    src_nodes = tf.random.uniform((BATCH_SIZE, INPUT_DIMS))
    neighbor_nodes = tf.random.uniform(
        (BATCH_SIZE * NEIGHBOR_SIZE, INPUT_DIMS))

    # test inputs for attention aggregator
    node_vectors = tf.random.uniform((BATCH_SIZE * 6, INPUT_DIMS))
    segment_sizes = random_sum_to(BATCH_SIZE * 6, BATCH_SIZE)
    src_indices = [0]
    for i in range(len(segment_sizes) - 1):
        start = src_indices[i]
        end = start + segment_sizes[i]
        src_indices.append(end)
    segment_sizes = tf.constant(segment_sizes)
    src_indices = tf.constant(src_indices)

    sage_layer = GraphSAGELayer(OUTPUT_DIMS, aggregate_method='mean')
    st = time.time()
    results = sage_layer((src_nodes, neighbor_nodes))
    ed1 = time.time()
    print(ed1 - st, 's')
