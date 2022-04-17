import tensorflow as tf

from tensorflow.keras.layers import Dropout, Dense


class FeatureExtractionLayer(tf.keras.layers.Layer):
    def __init__(self, output_dims, dropout=0., use_bias=True):
        super(FeatureExtractionLayer, self).__init__()
        self.output_dims = output_dims
        self.dropout_layer = Dropout(dropout)
        self.use_bias = use_bias

        self.inner_layer = Dense(4*self.output_dims, use_bias=self.use_bias)
        self.outer_layer = Dense(
            self.output_dims, activation='relu', use_bias=self.use_bias)

    def call(self, inputs, training=None):
        x = inputs
        # first dense layer
        x = self.inner_layer(x)
        x = self.dropout_layer(x, training=training)
        return self.outer_layer(x)


if __name__ == '__main__':
    # test
    import time

    # parameters
    BATCH_SIZE = 64
    INPUT_DIMS = 20
    OUTPUT_DIMS = 128
    # generate random test data
    inputs = tf.random.uniform(
        (BATCH_SIZE, INPUT_DIMS), dtype=tf.dtypes.float32)

    feature_extraction = FeatureExtractionLayer(
        OUTPUT_DIMS, dropout=0.5, use_bias=True)
    st = time.time()
    results = feature_extraction(inputs, training=True)
    ed = time.time()
    t = ed - st
    print(t, 's')
