### Function 모음 ###

def vector_add(v, w):
    return [v_i + w_i for v_i, w_i in zip(v,w)]

def dot(v,w):
    """
    내적
    """
    return sum(v_i*w_i for v_i, w_i in zip(v,w))
