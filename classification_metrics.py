import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score


def classification_eval_metrics(logits, labels, num_labels):
    predictions = logits.argmax(logits)

    info_dict = {
        "predictions": predictions,
        "labels": labels,
    }

    if len(self.predictions) == 0 or len(self.labels) == 0:
        tf.logging.info('empty data to evaluate')
        return {'accuracy': 0.0, 'micro_f1': 0.0,
                'macro_f1': 0.0, 'weighted_f1': 0.0}

    self.labels = np.stack(self.labels)
    self.predictions = np.stack(self.predictions)

    micro_f1 = f1_score(self.labels, self.predictions, labels=labels, average='micro')
    macro_f1 = f1_score(self.labels, self.predictions, labels=labels, average='macro')
    weighted_f1 = f1_score(self.labels, self.predictions, labels=labels, average='weighted')
    accuracy = accuracy_score(self.labels, self.predictions)
    return {'py_accuracy': accuracy, 'py_micro_f1': micro_f1,
            'py_macro_f1': macro_f1, 'py_weighted_f1': weighted_f1}

    label_idxs = [i for i in range(num_labels)]
    metric_dict = evaluator.get_metric_ops(info_dict, label_idxs)
    ret_metrics = evaluator.evaluate(label_idxs)

    tf.summary.scalar("accuracy", ret_metrics['py_accuracy'])
    tf.summary.scalar("micro_f1", ret_metrics['py_micro_f1'])
    tf.summary.scalar("macro_f1", ret_metrics['py_macro_f1'])
    tf.summary.scalar("weighted_f1", ret_metrics['py_weighted_f1'])
    return metric_dict
