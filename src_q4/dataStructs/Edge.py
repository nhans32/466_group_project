class Edge:
    def __init__(self, value, ghost, continuous):
        self.value = value
        self.ghost = ghost
        self.endNode = None
        self.continuous = continuous
        self.direction = None