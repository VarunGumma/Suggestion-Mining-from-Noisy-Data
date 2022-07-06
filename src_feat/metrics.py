import numpy as np
from re import finditer

PATTERN = "S|B+I*E+"
f = lambda S: ''.join(["BIOESX"[i] for i in S])

def _compute(y_true, y_pred, value="recall"):
    y_true, y_pred = f(y_true), f(y_pred)
    S = y_true if value == "recall" else y_pred
    indices = [(m.start(0), m.end(0)) for m in finditer(PATTERN, S)]
    retrieved_n_relevant = len([1 for (i, j) in indices if y_true[i:j] == y_pred[i:j]])
    return [retrieved_n_relevant, len(indices)]

# recall := how many relevent items have been retrieved ?
# precision := how many retrieved items are relevant ?
def metric(y_true, y_pred, value="recall"):
    s = np.sum([_compute(yt, yp, value=value) for (yt, yp) in zip(y_true, y_pred)], axis=0)
    return s[0] / (s[1] + 1e-8)

def f1score(y_true, y_pred):
    r = metric(y_true, y_pred, value="recall")
    p = metric(y_true, y_pred, value="precision")
    f1 = (2 * p * r)/(p + r + 1e-8)
    return r, p, f1