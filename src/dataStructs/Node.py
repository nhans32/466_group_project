class Node:
    def __init__(self, var, leaf):
        self.var = var
        self.children = []
        self.leaf = leaf
        self.p = None