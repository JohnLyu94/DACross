import pickle
import time
from datetime import datetime
import os

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from sklearn.metrics import f1_score
from model.chained_trip_net import ChainedTripNet
from model.baseline import BiLSTM
from utils.batch_generator import TripChainGenerator
from utils.strategy import NonImprovementEarlyStopping


def train_step(X, model):
    inputs, masks, indicies, time_en = X
    with tf.GradientTape() as tape:
        logits, log_loss = model(inputs, time_en, masks, indicies, training=False)
        loss = - tf.reduce_mean(log_loss)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss.numpy(), logits


def val_step(X, model):
    inputs, masks, indicies, time_en = X
    logits, log_loss = model(inputs, time_en, masks, indicies, training=False)
    loss = - tf.reduce_mean(log_loss)

    return loss.numpy(), logits


def crf_predict(y, logits, seq_length):
    # predict
    labels, _ = y
    flatten_labels = [label for seq in labels for label in seq]
    paths = []
    for logit, text_len in zip(logits, seq_length):
        viterbi_path, _ = tfa.text.viterbi_decode(logit[:text_len], model.aggregation_layer.transition_matrix)
        paths.append(viterbi_path)
    flatten_paths = [label for seq in paths for label in seq]
    return flatten_labels, flatten_paths


if __name__ == '__main__':
    # load data
    with open('./data/JP/tokyo_23_tel_train.pkl', 'rb') as f:
        train_set = pickle.load(f)
    with open('./data/JP/tokyo_23_tel_val.pkl', 'rb') as f:
        val_set = pickle.load(f)

    # global parameters
    BATCH_SIZE = 128
    DROPOUT = 0.1
    USE_BIAS = False
    NUM_CLASSES = 8
    LR_RATE = 1e-4
    MAX_EPOCHS = 300
    SHUFFLE = True
    SERVER = 'kusakabe_lab'
    DATASET = 'tokyo_23'

    train_generator = TripChainGenerator(train_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    val_generator = TripChainGenerator(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # local parameters
    params = {
        'embedding': {
            'model_dims': 512,
            'graph_dims': 512,
            'num_heads': 2,
            'activation': 'elu',
            'fusion_method': 'avg',
            'graph_model': 'gat'
        },
        'cross': {
            'model_dims': 512,
            'inner_dims': int(512 * 4 * 0.5),
            'num_heads': 4,
            'activation': 'elu',
            'pooling': 'mean'
        },
        'encoder_1': {
            'model_dims': 512,
            'dff': 512 * 1,
            'num_heads': 2,
            'num_layers': 6
        },
        'encoder_2': {
            'model_dims': 512,
            'dff': 512 * 1,
            'num_heads': 2,
            'num_layers': 4
        },
        'aggregation': {
            'rnn_units': 256,
        }
    }

    # init model
    model = ChainedTripNet(NUM_CLASSES, dropout=DROPOUT, crf_loss=True, use_bias=USE_BIAS, **params)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    checkpoint = tf.train.Checkpoint(ChainedTripNet=model)
    current_time = datetime.now().strftime('%Y%m%d%H%M%S_' + SERVER)
    log_path = './logs/' + current_time
    os.mkdir(log_path)
    manager = tf.train.CheckpointManager(checkpoint, log_path, checkpoint_name='ctn_%s.ckpt' % DATASET,
                                         max_to_keep=100)
    log_step = 50
    val_log_step = 50
    patience = 10
    es = NonImprovementEarlyStopping(patience=patience, delta=0)
    sum_epoch_time = []
    val_epoch_losses = []

    # init extreme values
    max_val_micro = float('-inf')
    max_val_macro = float('-inf')

    # train
    for epoch in range(MAX_EPOCHS):
        print("---***---***---***---EPOCH %d---***---***---***---" % (epoch + 1))
        train_losses = []
        val_losses = []
        y_train_pred = []
        y_train_true = []
        y_val_pred = []
        y_val_true = []

        train_batches = iter(train_generator)
        val_batches = iter(val_generator)
        step_length = train_batches.num_steps
        val_step_length = val_batches.num_steps

        # train epoch
        sum_step_time = []
        st_epoch = time.time()
        for step, batch_inputs in enumerate(train_batches):
            st_step = time.time()
            inputs, masks, indicies, time_en, y = batch_inputs
            X = (inputs, masks, indicies, time_en)
            step_loss, logits = train_step(X, model)
            train_losses.append(step_loss)

            flatten_labels, flatten_paths = crf_predict(y, logits, indicies[0])
            train_micro_f1 = f1_score(flatten_labels, flatten_paths, average='micro')  # step accuracy
            train_macro_f1 = f1_score(flatten_labels, flatten_paths, average='macro')

            y_train_true.append(flatten_labels)  # for calculating metrics on the whole training set
            y_train_pred.append(flatten_paths)

            ed_step = time.time()
            sum_step_time.append(ed_step - st_step)
            if (step + 1) % log_step == 0:
                print(
                    "TRAIN_STEP[%d/%d] in %.4fs | ETA: %.4fs | train_loss: %.4f "
                    "train_f1_micro: %.4f train_f1_macro: %.4f" %
                    (
                        step + 1, step_length, ed_step - st_step,
                        (step_length - step - 1) * np.mean(sum_step_time),
                        train_losses[-1], train_micro_f1, train_macro_f1
                    )
                )

        # val epoch
        for step, batch_inputs in enumerate(val_batches):
            inputs, masks, indicies, time_en, y = batch_inputs
            X = (inputs, masks, indicies, time_en)
            step_loss, logits = val_step(X, model)
            val_losses.append(step_loss)

            flatten_labels, flatten_paths = crf_predict(y, logits, indicies[0])
            val_micro_f1 = f1_score(flatten_labels, flatten_paths, average='micro')  # step accuracy
            val_macro_f1 = f1_score(flatten_labels, flatten_paths, average='macro')

            y_val_true.append(flatten_labels)
            y_val_pred.append(flatten_paths)

            if (step + 1) % val_log_step == 0:
                print(
                    "VAL_STEP[%d/%d] | val_loss: %.4f val_f1_micro: %.4f val_f1_macro: %.4f" %
                    (
                        step + 1, step_length,
                        val_losses[-1], val_micro_f1, val_macro_f1
                    )
                )

        ed_epoch = time.time()
        sum_epoch_time.append(ed_epoch - st_epoch)
        # end of epoch
        # epoch metric
        y_train_pred = [y for step in y_train_pred for y in step]
        y_train_true = [y for step in y_train_true for y in step]
        y_val_pred = [y for step in y_val_pred for y in step]
        y_val_true = [y for step in y_val_true for y in step]

        train_micro_f1 = f1_score(y_train_true, y_train_pred, average='micro')
        train_macro_f1 = f1_score(y_train_true, y_train_pred, average='macro')
        val_micro_f1 = f1_score(y_val_true, y_val_pred, average='micro')
        val_macro_f1 = f1_score(y_val_true, y_val_pred, average='macro')

        avg_train_losses = np.mean(train_losses)
        avg_val_losses = np.mean(val_losses)
        es.__call__(avg_train_losses, avg_val_losses)

        print(
            "EPOCH[%d/%d] in %.4fs | ETA: %.4fs "
            "| train_loss: %.4f train_f1_micro: %.4f train_f1_macro: %.4f "
            "| val_loss: %.4f val_f1_micro: %.4f val_f1_macro: %.4f" %
            (
                epoch + 1, MAX_EPOCHS, ed_epoch - st_epoch,
                (MAX_EPOCHS - epoch - 1) * np.mean(sum_epoch_time),
                float(avg_train_losses), train_micro_f1, train_macro_f1,
                float(avg_val_losses), val_micro_f1, val_macro_f1
            )
        )
        manager.save()

        if max_val_macro < val_macro_f1:
            max_val_macro = val_macro_f1
        if max_val_micro < val_micro_f1:
            max_val_micro = val_micro_f1

        # early stopping
        if es.judge():
            break
