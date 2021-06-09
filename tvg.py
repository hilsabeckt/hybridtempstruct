import math
from sortedcontainers import SortedDict


class Edge:
    def __init__(self, u, v, begin=None, end=None, **attr):
        self.u = u
        self.v = v
        self.begin = begin
        self.end = end
        self.attr = attr

    def __repr__(self):
        return (self.u, self.v, self.begin, self.end, self.attr).__str__()


class TVG:
    def __init__(self):
        self.node = {}
        self.adj = SortedDict()
        self.begin = math.inf
        self.end = -math.inf

    def add_edge(self, u, v, begin, end=None):
        if end is None:
            end = begin
        edge = Edge(u, v, begin, end)
        self.node.setdefault(u, {})
        self.node.setdefault(v, {})
        self.adj.setdefault(begin, SortedDict()).setdefault(end, {}).setdefault(u, {}).setdefault(v, []).append(edge)
        self.adj.setdefault(begin, SortedDict()).setdefault(end, {}).setdefault(v, {}).setdefault(u, []).append(edge)

        self.begin = min(self.begin, begin)
        self.end = max(self.end, end)

    def slice(self, begin=None, end=None):
        if begin is None:
            begin = self.begin
        if end is None:
            end = self.end

        edges = set()
        for b in self.adj.irange(begin, end, inclusive=(True, False)):
            for e in self.adj[b].irange(b, end, inclusive=(True, False)):
                for u in self.adj[b][e]:
                    for v in self.adj[b][e][u]:
                        for edge in self.adj[b][e][u][v]:
                            if edge not in edges:
                                yield edge
                                edges.add(edge)

    def interval(self):
        return self.begin, self.end
