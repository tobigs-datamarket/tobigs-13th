### Function 모음 ###

def vector_add(v, w):
    return [v_i + w_i for v_i, w_i in zip(v,w)]

def dot(v,w):
    """
    내적
    """
    return sum(v_i*w_i for v_i, w_i in zip(v,w))

def neg(f):
    """모든 인풋 x에 대해 -f(x)를 리턴함"""
    return lambda *args, **kwargs: -f(*args, **kwargs)


def neg_all(f):
    """f가 리스트를 리턴할 때 y의 모든 함수를 음수값으로 리턴할 때"""
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]
