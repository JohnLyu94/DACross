import tensorflow as tf


class SequenceEval(tf.keras.metrics.Metric):
    def __init__(self, n_classes, name='sequence_evaluation', from_logits=None, **kwargs):
        super(SequenceEval, self).__init__(name=name, **kwargs)
        self.confusion_matrix = self.add_weight(shape=(n_classes, n_classes),
                                                name='confusion_matrix',
                                                initializer='zeros',
                                                dtype=tf.int64)
        self.tp = self.add_weight(shape=(n_classes,),
                                  name='true_positives',
                                  initializer='zeros',
                                  dtype=tf.int64)
        self.fn = self.add_weight(shape=(n_classes,),
                                  name='false_negatives',
                                  initializer='zeros',
                                  dtype=tf.int64)
        self.fp = self.add_weight(shape=(n_classes,),
                                  name='false_positives',
                                  initializer='zeros',
                                  dtype=tf.int64)
        self.from_logits = from_logits
        self.n_classes = n_classes

    def update_state(self, y_true, y_pred, weights=None):
        if self.from_logits:
            y_pred = tf.argmax(y_pred, axis=1)

        cm = tf.math.confusion_matrix(y_true, y_pred, self.n_classes, weights=weights, dtype=tf.int64)
        # tp
        self.tp.assign_add(tf.linalg.diag_part(cm))
        # fp
        self.fp.assign_add(tf.math.subtract(tf.reduce_sum(cm, axis=0), tf.linalg.diag_part(cm)))
        # fn
        self.fn.assign_add(tf.math.subtract(tf.reduce_sum(cm, axis=1), tf.linalg.diag_part(cm)))
        self.confusion_matrix.assign_add(cm)

    def result(self):
        # macro
        categorical_precision = tf.divide(self.tp, tf.add_n([self.tp, self.fp]))
        categorical_recall = tf.divide(self.tp, tf.add_n([self.tp, self.fn]))
        macro_f1 = tf.divide(tf.multiply(2, tf.multiply(categorical_recall, categorical_precision)), tf.add_n(
            [categorical_precision, categorical_recall]))
        macro_f1 = tf.where(tf.math.is_nan(macro_f1), [0.] * self.n_classes, macro_f1)
        macro_f1 = tf.reduce_mean(macro_f1)
        # micro
        precision = tf.reduce_sum(self.tp) / tf.add_n([tf.reduce_sum(self.tp), tf.reduce_sum(self.fp)])
        recall = tf.reduce_sum(self.tp) / tf.add_n([tf.reduce_sum(self.tp), tf.reduce_sum(self.fn)])
        micro_f1 = (2 * precision * recall) / (precision + recall)

        return [micro_f1, macro_f1]

    def reset_states(self):
        self.confusion_matrix.assign(tf.zeros((self.n_classes, self.n_classes), dtype=tf.int64))
        self.tp.assign(tf.zeros((self.n_classes,), dtype=tf.int64))
        self.fn.assign(tf.zeros((self.n_classes,), dtype=tf.int64))
        self.fp.assign(tf.zeros((self.n_classes,), dtype=tf.int64))


if __name__ == '__main__':
    import numpy as np

    n_classes = 3
    y_true = np.array([2, 0, 1, 2])
    y_pred = np.array([
        [0.1, 0.2, 0.7],
        [0.3, 0.5, 0.2],
        [0.4, 0.3, 0.3],
        [0.2, 0.2, 0.6]
    ])  # [2,1,0,2]
    y_pred_label = np.array([2, 1, 0, 2])
    metric = SequenceEval(n_classes, from_logits=True)
    metric.update_state(y_true, y_pred)
    print(metric.result())
