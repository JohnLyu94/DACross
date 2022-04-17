import json
import pickle
import time

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from sklearn.metrics import f1_score, classification_report
from model.chained_trip_net import ChainedTripNet, ChainedTripNet_A, ChainedTripNet_B, ChainedTripNet_C, \
    ChainedTripNet_E
from model.baseline import BiLSTM_FUSE, ESIM, BiLSTM
from utils.batch_generator import TripChainGenerator


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
    file_name = './data/JP/tokyo_23_test.pkl'
    with open(file_name, 'rb') as f:  # warning!
        val_set = pickle.load(f)
    print(file_name)

    # val_set['chains'] = [val_set['chains'][478]]

    # global parameters
    BATCH_SIZE = 1
    DROPOUT = 0.1
    USE_BIAS = True
    NUM_CLASSES = 8  # warning!

    # local parameters
    with open("./conditions/tbi_hyper_parameters.json", "r") as f:
        params = json.load(f)

    val_generator = TripChainGenerator(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # load model
    model = ChainedTripNet(NUM_CLASSES, crf_loss=True, use_bias=USE_BIAS, **params)
    checkpoint = tf.train.Checkpoint(ChainedTripNet=model)

    checkpoint.restore('./path-to-ckpt')  # full

    sum_epoch_time = []
    val_epoch_losses = []
    val_losses = []

    val_batches = iter(val_generator)
    val_step_length = val_batches.num_steps
    micro_f1 = []
    macro_f1 = []
    classification_reports = {}
    classification_reports['detail'] = []
    classification_reports['y_pred'] = []
    labels = list(range(8))

    for i in range(1):
        # val epoch
        for step, batch_inputs in enumerate(val_batches):
            st = time.time()
            inputs, masks, indicies, time_en, y = batch_inputs
            X = (inputs, masks, indicies, time_en)
            step_loss, logits = val_step(X, model)
            val_losses.append(step_loss)

            flatten_labels, flatten_paths = crf_predict(y, logits, indicies[0])
            if step == 0:
                y_true = flatten_labels
                y_pred = flatten_paths
            else:
                y_true += flatten_labels
                y_pred += flatten_paths

            ed = time.time()
            print(
                "VAL_STEP[%d/%d] in %fs | val_loss: %.4f" %
                (
                    step + 1, val_step_length, ed - st, val_losses[-1]
                )
            )
        micro_f1.append(f1_score(y_true, y_pred, average='micro'))
        macro_f1.append(f1_score(y_true, y_pred, average='macro'))
        classification_reports['detail'].append(
            classification_report(y_true, y_pred, labels=labels, target_names=val_set["others"]['classes'],
                                  digits=4, output_dict=True))
        classification_reports['y_pred'] = y_pred

    print(np.mean(micro_f1), np.std(micro_f1))
    print(np.mean(macro_f1), np.std(macro_f1))

    """
    # with open("./results/tokyo_23_tel_sparse_no_scaler_reports.pkl", "wb") as f:
    #    pickle.dump(classification_reports, f)
    """
