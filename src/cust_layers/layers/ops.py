import tensorflow.keras.backend as K

def l2normalize(v, eps=1e-12):
    return v / (K.sum(v **2) ** 0.5 + eps)

def power_iteration(W,u):
    _u = u
    _v = l2normalize(K.dot(_u, K.transpose(W)))
    _u = l2normalize(K.dot(_v,W))
    return _u, _v

