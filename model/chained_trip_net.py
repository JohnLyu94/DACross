import tensorflow as tf
import numpy as np
from .layers.aggregator import AttentionAggregator
from .layers.mlp import FeatureExtractionLayer
from .layers.attention import *
from .layers.rnn import ChildSumTreeLSTMCell
from utils.batch_generator import TripChainGenerator
from tensorflow_addons.text import crf_log_likelihood
import tensorflow_addons as tfa
from sklearn.metrics import f1_score
from utils.metrics import SequenceEval
import math


def cyclical_feat(input, mask):
    sin_arrive = tf.expand_dims(tf.multiply(tf.math.sin(2 * math.pi * input / (3600 * 24)), mask), -1)
    cos_arrive = tf.expand_dims(tf.multiply(tf.math.cos(2 * math.pi * input / (3600 * 24)), mask), -1)

    return tf.cast(tf.concat([sin_arrive, cos_arrive], axis=2), tf.float32)


def reconstruct_from_graph(graph_embeddings, region_mask, index_array):
    region_mask = tf.transpose(region_mask, perm=[1, 0])
    # re-construct zones
    region_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for i, region_seq in enumerate(index_array):
        region_seq = tf.boolean_mask(
            region_seq, tf.cast(region_seq, tf.bool)) - 1
        region_seq = tf.gather(graph_embeddings, region_seq)
        # padding to same shape with 0
        # (MAX_LENGTH, GRAPH_OUTPUT_DIMS)
        # which means [-1,:] should be fulfilled with 0
        region_seq = tf.pad(region_seq, tf.constant(
            [[0, region_mask.shape[0] - region_seq.shape[0]], [0, 0]]), 'CONSTANT')
        region_array = region_array.write(i, region_seq)
    # concat and release array
    # (max_length, batch_size, input_dims)
    region_embeddings = region_array.stack()
    region_array = region_array.close()

    return region_embeddings


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self,
                 dropout=0.,
                 use_bias=False,
                 **kwargs):
        super(EmbeddingLayer, self).__init__()
        output_dims = kwargs['model_dims']
        self.graph_model = kwargs['graph_model']
        assert self.graph_model in ['gat', 'mlp']

        self.behavior_embedding_layer = FeatureExtractionLayer(
            output_dims, dropout=dropout, use_bias=use_bias)

        self.time_map = Dense(output_dims)

        if self.graph_model == 'mlp':
            self.consequence_embedding_layer = FeatureExtractionLayer(
                output_dims, dropout=dropout, use_bias=use_bias)

        elif self.graph_model == 'gat':
            graph_dims = kwargs['graph_dims']
            num_heads = kwargs['num_heads']
            activation = kwargs['activation']
            fusion_mehtod = kwargs['fusion_method']

            if fusion_mehtod == 'avg':
                assert graph_dims == output_dims
            else:
                assert graph_dims * num_heads == output_dims
            self.graph_layer = AttentionAggregator(
                graph_dims,
                num_heads=num_heads,
                dropout=dropout,
                use_bias=use_bias,
                activation=activation,
                fusion_method=fusion_mehtod
            )

    def call(self, inputs, masks, time_en, seq_idx, training=None):
        region_batch, trip_batch = inputs
        trip_mask, region_mask, graph_mask = masks
        arrive, depart = time_en
        region_idx_array = seq_idx

        arrive = tf.cast(arrive, tf.float32)
        depart = tf.cast(depart, tf.float32)

        # (BATCH_SIZE, MAX_LEN_2, OUTPUT_DIMS)
        trip_embeddings = self.behavior_embedding_layer(
            trip_batch, training=training)

        if self.graph_model == 'mlp':
            # (BATCH_SIZE, MAX_LEN_1, OUTPUT_DIMS)
            region_embeddings = self.consequence_embedding_layer(
                region_batch, training=training)

        elif self.graph_model == 'gat':
            # (BATCH_SIZE, MAX_LEN_1, OUTPUT_DIMS)
            graph_embeddings = self.graph_layer(
                region_batch, graph_mask, training=training)

            # reconstruct batch
            region_embeddings = reconstruct_from_graph(graph_embeddings, region_mask, region_idx_array)

        arrive_feat = cyclical_feat(arrive, region_mask)
        depart_feat = cyclical_feat(depart, trip_mask)

        region_embeddings = region_embeddings + self.time_map(arrive_feat)
        trip_embeddings = trip_embeddings + self.time_map(depart_feat)

        return region_embeddings, trip_embeddings


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 dropout=0.,
                 **kwargs):
        super(Encoder, self).__init__()

        dff = kwargs['dff']
        num_heads = kwargs['num_heads']
        self.d_model = kwargs['model_dims']
        self.num_layers = kwargs['num_layers']

        self.enc_layers = [EncoderLayer(self.d_model, dff, num_heads, dropout=dropout)
                           for _ in range(self.num_layers)]

        self.dropout = Dropout(dropout)

    def call(self, x, mask, training=None):
        seq_len = tf.shape(x)[1]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask, training=training)

        return x  # (batch_size, input_seq_len, d_model)


class CrossLayer(CoAttentionFusion):
    def __init__(self,
                 dropout=0.,
                 use_bias=True,
                 **kwargs):
        model_dims = kwargs['model_dims']
        inner_dims = kwargs['inner_dims']
        num_heads = kwargs['num_heads']
        activation = kwargs['activation']
        pooling = kwargs['pooling']

        super(CrossLayer, self).__init__(model_dims,
                                         inner_dims,
                                         multi_head_num=num_heads,
                                         dropout=dropout,
                                         activation=activation,
                                         use_bias=use_bias,
                                         pooling=pooling)


class AggregationLayer(tf.keras.layers.Layer):
    def __init__(self,
                 num_classes,
                 use_bias=False,
                 dropout=0.,
                 crf_loss=True,
                 **kwargs):
        super(AggregationLayer, self).__init__()
        self.rnn_units = kwargs['rnn_units']
        self.crf_loss = crf_loss
        self.recurrent_cell = ChildSumTreeLSTMCell(self.rnn_units, use_bias=use_bias, dropout=dropout,
                                                   recurrent_dropout=dropout)

        self.classifier = Dense(num_classes, use_bias=use_bias)
        self.cell_children = self.add_weight(shape=(2, self.rnn_units), name='chidren_init_cells')

        if self.crf_loss:
            self.transition_matrix = tf.Variable(tf.random.uniform(shape=(num_classes, num_classes)),
                                                 name='transition_matrix')

    def call(self, inputs, mask, indicies, training=None):
        trip_hiddens, region_hiddens = inputs
        trip_mask = mask
        tag_indicies, sequence_lengths = indicies

        batch_size, max_len, _ = tf.shape(trip_hiddens)

        unstack_trip = tf.unstack(trip_hiddens, axis=1)
        unstack_region = tf.unstack(region_hiddens, axis=1)

        for i in range(max_len):
            parents = unstack_trip[i]
            children_l = tf.expand_dims(unstack_region[i], 0)
            children_r = tf.expand_dims(unstack_region[i + 1], 0)
            children = tf.concat([children_l, children_r], axis=0)

            output_hiddens, _ = self.recurrent_cell(parents, [children,
                                                              tf.repeat(tf.expand_dims(self.cell_children, 1),
                                                                        repeats=batch_size, axis=1)
                                                              ], training=training)
            logits = self.classifier(output_hiddens)

            if not self.crf_loss:
                logits = tf.nn.softmax(logits, axis=1)

            if i == 0:
                output_logits = tf.expand_dims(logits, 1)
            else:
                output_logits = tf.concat([output_logits, tf.expand_dims(logits, 1)], axis=1)

        if self.crf_loss:
            loglikelihood, self.transition_matrix = crf_log_likelihood(
                output_logits,
                tag_indicies,
                sequence_lengths,
                self.transition_matrix
            )
            return output_logits, loglikelihood

        else:
            unstack_logits = tf.unstack(output_logits, axis=0)
            unstack_trip_mask = tf.unstack(trip_mask, axis=0)
            # drop masks (TOTAL_TRIPS,NUM_CLASSES)
            for i in range(batch_size):
                if i == 0:
                    outputs = tf.boolean_mask(unstack_logits[i], unstack_trip_mask[i])
                else:
                    outputs = tf.concat([outputs,
                                         tf.boolean_mask(unstack_logits[i], unstack_trip_mask[i])], axis=0)
            return outputs


class ChainedTripNet(tf.keras.models.Model):
    def __init__(self,
                 num_classes,
                 dropout=0.,
                 use_bias=False,
                 crf_loss=False,
                 **kwargs):
        super(ChainedTripNet, self).__init__()
        embedding_params = kwargs['embedding']
        self.embedding_model = embedding_params['graph_model']
        encoder_1_params = kwargs['encoder_1']
        encoder_2_params = kwargs['encoder_2']
        cross_params = kwargs['cross']
        aggregation_params = kwargs['aggregation']

        self.embedding_layer = EmbeddingLayer(dropout=dropout, use_bias=use_bias, **embedding_params)
        self.encoder_layer_1 = Encoder(dropout=dropout, **encoder_1_params)
        self.encoder_layer_2 = Encoder(dropout=dropout, **encoder_2_params)
        self.cross_layer = CrossLayer(dropout=dropout, use_bias=use_bias, **cross_params)
        self.aggregation_layer = AggregationLayer(num_classes, use_bias=use_bias, dropout=dropout, crf_loss=crf_loss,
                                                  **aggregation_params)

    def call(self, inputs, time_feats, masks, indicies, training=None):
        trip_tensor, region_tensor, graph_tensor = inputs
        trip_mask, region_mask, graph_mask = masks
        sequence_lengths, tag_indicies, region_idx_array = indicies

        if self.embedding_model == 'gat':
            region_embeddings, trip_embeddings = self.embedding_layer([graph_tensor, trip_tensor], masks, time_feats,
                                                                      region_idx_array, training=training)
        elif self.embedding_model == 'mlp':
            region_embeddings, trip_embeddings = self.embedding_layer([region_tensor, trip_tensor], masks, time_feats,
                                                                      region_idx_array, training=training)

        trip_hiddens = self.encoder_layer_1(trip_embeddings, trip_mask, training=training)
        region_hiddens = self.encoder_layer_1(region_embeddings, region_mask, training=training)

        trip_crossed, region_crossed = self.cross_layer([trip_hiddens, region_hiddens], [trip_mask, region_mask],
                                                        training=training)

        trip_composed = self.encoder_layer_2(trip_crossed, trip_mask, training=training)
        region_composed = self.encoder_layer_2(region_crossed, region_mask, training=training)

        outputs = self.aggregation_layer([trip_composed, region_composed], trip_mask, [tag_indicies, sequence_lengths])

        return outputs