import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout


def init_state(batch_size, output_dims):
    return tf.zeros((batch_size, output_dims))


def child_segment(n_root, n_child):
    neighbor_node_idxs = tf.tile(tf.range(n_root), [n_child // n_root])
    return neighbor_node_idxs, n_root


class LSTMCell(tf.keras.layers.Layer):
    def __init__(self,
                 output_dims,
                 use_bias=False,
                 dropout=0.,
                 recurrent_dropout=0.):
        super(LSTMCell, self).__init__()
        self.output_dims = output_dims

        # input gate
        self.input_gate = Dense(output_dims, use_bias=use_bias)
        self.input_gate_u = Dense(output_dims, use_bias=False)

        self.forget_gate = Dense(output_dims, use_bias=use_bias)
        self.forget_gate_u = Dense(output_dims, use_bias=False)

        self.output_gate = Dense(output_dims, use_bias=use_bias)
        self.output_gate_u = Dense(output_dims, use_bias=False)

        # u
        self.u = Dense(output_dims, use_bias=use_bias)
        self.u_u = Dense(output_dims, use_bias=False)

        self.dropout = Dropout(dropout)
        self.recurrent_dropout = Dropout(recurrent_dropout)

    def call(self, inputs, states, training=None):
        x_t = inputs
        hidden_t_1, c_t_1 = states

        x_t = self.dropout(x_t, training=training)
        hidden_t_1 = self.recurrent_dropout(hidden_t_1, training=training)

        # input gate
        input_gate = tf.nn.sigmoid(self.input_gate(x_t) + self.input_gate_u(hidden_t_1))
        # forget gate
        forget_gate = tf.nn.sigmoid(self.forget_gate(x_t) + self.forget_gate_u(hidden_t_1))
        # output gate
        output_gate = tf.nn.sigmoid(self.output_gate(x_t) + self.output_gate_u(hidden_t_1))
        # u
        u = tf.nn.tanh(self.u(x_t) + self.u_u(hidden_t_1))
        # cell state
        c_t = tf.math.multiply(input_gate, u) + \
              tf.math.multiply(forget_gate, c_t_1)
        # output->hidden_t
        hidden_t = tf.math.multiply(output_gate, tf.nn.tanh(c_t))

        return hidden_t, c_t


class ChildSumTreeLSTMCell(LSTMCell):
    def __init__(self,
                 output_dims,
                 use_bias=True,
                 dropout=0.,
                 recurrent_dropout=0.):
        super(ChildSumTreeLSTMCell, self).__init__(output_dims, use_bias=use_bias, dropout=dropout,
                                                   recurrent_dropout=recurrent_dropout)

    def call(self, inputs, states, training=None):
        # (BATCH_SIZE, DIMS)
        x_t = inputs

        # (NUM_CHILDREN, BATCH_SIZE, OUTPUT_DIM)
        child_hiddens, child_cells = states

        x_t = self.dropout(x_t, training=training)
        child_sum_hiddens = tf.math.reduce_sum(child_hiddens, axis=0)
        child_sum_hiddens = self.recurrent_dropout(child_sum_hiddens, training=training)

        input_gate = tf.nn.sigmoid(self.input_gate(x_t) + self.input_gate_u(child_sum_hiddens))
        output_gate = tf.nn.sigmoid(self.output_gate(x_t) + self.output_gate_u(child_sum_hiddens))
        u = tf.nn.tanh(self.u(x_t) + self.u_u(child_sum_hiddens))

        forget_gate = tf.nn.sigmoid(self.forget_gate(x_t) + self.forget_gate_u(child_hiddens))

        c_t = tf.math.multiply(input_gate, u) + \
              tf.reduce_sum(tf.math.multiply(forget_gate, child_cells), axis=0)
        hidden_t = tf.math.multiply(output_gate, tf.nn.tanh(c_t))

        return hidden_t, c_t


if __name__ == '__main__':
    # test
    import time

    BATCH_SIZE = 64
    INPUT_DIMS = 20
    OUTPUT_DIMS = 128
    NUM_CHILDREN = 2
    # generate random test data
    x_t = tf.random.uniform(
        (BATCH_SIZE, INPUT_DIMS), dtype=tf.dtypes.float32)
    state_t_1 = tf.random.uniform(
        (BATCH_SIZE, OUTPUT_DIMS), dtype=tf.dtypes.float32)
    c_t_1 = tf.random.uniform(
        (BATCH_SIZE, OUTPUT_DIMS), dtype=tf.dtypes.float32)

    lstm_cell = LSTMCell(OUTPUT_DIMS, use_bias=True)
    st = time.time()
    hidden_t, c_t = lstm_cell(x_t, [state_t_1, c_t_1])
    ed = time.time()
    print(ed - st, 's')

    child_hiddens = tf.random.uniform(
        (NUM_CHILDREN, BATCH_SIZE, OUTPUT_DIMS), dtype=tf.dtypes.float32)
    child_c_t_1 = tf.random.uniform(
        (NUM_CHILDREN, BATCH_SIZE, OUTPUT_DIMS), dtype=tf.dtypes.float32)

    sum_tree = ChildSumTreeLSTMCell(OUTPUT_DIMS, use_bias=True)

    st = time.time()
    hidden_t, c_t = sum_tree(x_t, [child_hiddens, child_c_t_1])
    ed = time.time()
    print(ed - st, 's')
