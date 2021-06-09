import math
from bisect import bisect_left
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


class Node:

    def __init__(self, low, high, edges=None):
        self.low = low
        self.high = high
        if edges is None:
            self.edges = []
        else:
            self.edges = edges

    def __repr__(self):
        return (self.low, self.high, len(self.edges)).__str__()

    def inInterval(self, begin, end):
        return (self.low < end and self.high > begin) or self.low == begin


class SnapshotTree:

    def __init__(self):
        self.nodes = SortedDict()
        self.begin = math.inf
        self.end = -math.inf
        self.number_of_edges = 0

    def __getitem__(self, item):
        start = item.start
        stop = item.stop
        if start is None:
            start = self.begin
        if stop is None:
            stop = self.end
        return self.slice(start, stop)

    def _add(self, timestamp):
        if timestamp not in self.nodes:
            self.nodes[timestamp] = Node(timestamp, timestamp)
            index = self.nodes.index(timestamp)
            if index >= 1:
                left_key, left_node = self.nodes.peekitem(index - 1)
                for edge in left_node.edges:
                    if edge.end > timestamp:
                        self.nodes[timestamp].edges.append(edge)

            if index <= len(self.nodes) - 2:
                right_key, right_node = self.nodes.peekitem(index + 1)
                for edge in right_node.edges:
                    if edge.begin <= timestamp and edge not in self.nodes[timestamp].edges:
                        self.nodes[timestamp].edges.append(edge)

    def add(self, edge):
        self._add(edge.begin)
        self._add(edge.end)

        if edge.begin == edge.end:
            self.nodes[edge.begin].edges.append(edge)
        else:
            for event in self.nodes.irange(edge.begin, edge.end, (True, False)):
                self.nodes[event].edges.append(edge)

        self.number_of_edges += 1
        self.begin = min(edge.begin, self.begin)
        self.end = max(edge.end, self.end)

    def add_from(self, edges):
        for edge in edges:
            self.nodes.setdefault(edge.begin, Node(edge.begin, edge.begin))
            self.nodes.setdefault(edge.end, Node(edge.end, edge.end))
            self.number_of_edges += 1
            self.begin = min(edge.begin, self.begin)
            self.end = max(edge.end, self.end)

        for edge in edges:
            if edge.begin == edge.end:
                self.nodes[edge.begin].edges.append(edge)
            else:
                for event in self.nodes.irange(edge.begin, edge.end, (True, False)):
                    self.nodes[event].edges.append(edge)

    def slice(self, begin, end):
        edges = set()
        events = list(self.nodes.keys())
        smallest_event = events[max(0, bisect_left(events, begin) - 1)]
        for node in self.nodes.irange(smallest_event, end, inclusive=(True, False)):
            for edge in self.nodes[node].edges:
                if edge not in edges:
                    yield edge
                    edges.add(edge)


class SnapshotGraph:
    def __init__(self, **attr):
        self.node = {}
        self.adj = {}
        self.tree = SnapshotTree()
        self.root = None
        self.attr = attr
        self.begin = math.inf
        self.end = -math.inf
        self.number_of_edges = 0

    def add_edges_from(self, edges):
        edges = [Edge(*e) for e in edges]
        self.tree.add_from(edges)
        self.begin = self.tree.begin
        self.end = self.tree.end

    def add_edge(self, u, v, begin, end):
        self.node.setdefault(u, {})
        self.node.setdefault(v, {})
        self.adj.setdefault(u, {}).setdefault(v, []).append((u, v, begin, end))
        self.adj.setdefault(v, {}).setdefault(u, []).append((u, v, begin, end))

        self.tree.add(Edge(u, v, begin, end))

        self.begin = min(self.begin, begin)
        self.end = max(self.end, end)
        self.number_of_edges += 1

    def interval(self):
        return self.begin, self.end

    def slice(self, interval_start, interval_end):
        yield from self.tree.slice(interval_start, interval_end)
