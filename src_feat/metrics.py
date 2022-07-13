import numpy as np
from re import finditer

PATTERN = "S|B+I*E+"
f = lambda S: ''.join(["BIOES<>$"[i] for i in S])

def _compute(y_true, y_pred, value="recall"):
    y_true, y_pred = f(y_true), f(y_pred)
    S = y_true if value == "recall" else y_pred
    indices = [(m.start(0), m.end(0)) for m in finditer(PATTERN, S)]
    retrieved_n_relevant = len([1 for (i, j) in indices if y_true[i:j] == y_pred[i:j]])
    return [retrieved_n_relevant, len(indices)]

# recall := how many relevent items have been retrieved ?
# precision := how many retrieved items are relevant ?
def _generic_metric(y_true, y_pred, value="recall"):
    s = np.sum([_compute(yt, yp, value=value) for (yt, yp) in zip(y_true, y_pred)], axis=0)
    return s[0] / (s[1] + 1e-8)

def recall(y_true, y_pred):
    return _generic_metric(y_true, y_pred, value="recall")

def precision(y_true, y_pred):
    return _generic_metric(y_true, y_pred, value="precision")

def f1score(y_true, y_pred, return_precision_recall=True):
    r = recall(y_true, y_pred)
    p = precision(y_true, y_pred)
    f1 = (2 * p * r)/(p + r + 1e-8)
    return (r, p, f1) if return_precision_recall else f1