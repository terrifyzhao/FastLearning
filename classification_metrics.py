from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from logger import logger
from tensorboardX import SummaryWriter
import numpy as np

writer = SummaryWriter(comment='model')


def binary_eval_metrics(logits, labels, batch=-1):
    predictions = np.argmax(logits, axis=1)

    if len(logits) == 0 or len(labels) == 0:
        logger.error('empty data to evaluate')
        return {'accuracy': 0.0, 'f1': 0.0, 'auc': 0.0}

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    auc = roc_auc_score(labels, [l[1] for l in logits])

    if batch >= 0:
        writer.add_scalar('accuracy', accuracy, global_step=batch)
        writer.add_scalar('f1', f1, global_step=batch)
        writer.add_scalar('auc', auc, global_step=batch)

    return {'accuracy': accuracy, 'f1': f1, 'auc': auc}


def multi_class_eval_metrics(logits, labels, batch=-1, average='micro'):
    if np.ndim(labels) != 2:
        raise ValueError('logits shape must be (n_samples, n_classes)')
    if len(logits) == 0 or len(labels) == 0:
        logger.info('empty data to evaluate')
        return {'accuracy': 0, 'f1': 0, 'auc': 0}

    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average=average)
    auc = roc_auc_score(labels, logits, average=average)

    if batch >= 0:
        writer.add_scalar('accuracy', accuracy, global_step=batch)
        writer.add_scalar('f1', f1, global_step=batch)
        writer.add_scalar('auc', auc, global_step=batch)

    return {'accuracy': accuracy, 'f1': f1, 'auc': auc}


if __name__ == '__main__':
    a = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.3, 0.7]])
    bs = [[1, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 1]]
    for i in range(5):
        b = bs[i]
        print(binary_eval_metrics(a, b, batch=i))
