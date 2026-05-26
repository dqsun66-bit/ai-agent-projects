# 全局向量库存储，所有线程共享
_vs = None

def set_vs(vs):
    global _vs
    _vs = vs

def get_vs():
    return _vs