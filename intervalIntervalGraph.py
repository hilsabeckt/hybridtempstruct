from sortedcontainers import SortedDict
import math


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

    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.max = high
        self.edges = []

        self.left = None
        self.right = None
        self.height = 0

    def __repr__(self):
        return (self.low, self.high, len(self.edges)).__str__()

    def __add__(self, other):
        parent_node = Node(min(self.low, other.low), max(self.high, other.high))

        return parent_node

    def inInterval(self, begin, end):
        return (self.low < end and self.high > begin) or self.low == begin


class IntervalTree:

    def __init__(self):
        self.nodes = {}
        self.root = None
        self.number_of_edges = 0

    def updateMax(self, node):

        if node.right and node.left:
            node.max = max(node.high, node.right.max, node.left.max)
        elif node.left:
            node.max = max(node.high, node.left.max)
        elif node.right:
            node.max = max(node.high, node.right.max)
        else:
            node.max = node.high

    def inOrder(self, root):

        if not root:
            return

        yield from self.inOrder(root.left)
        yield root
        yield from self.inOrder(root.right)

    def insert(self, root, node):
        if root is None:
            return node

        if node.low < root.low or (node.low == root.low and node.high < root.high):
            root.left = self.insert(root.left, node)
        else:
            root.right = self.insert(root.right, node)

        root.height = 1 + max(self.getHeight(root.left), self.getHeight(root.right))

        balance = self.getBalance(root)

        if balance > 1:
            if root.left.right == node:
                root.left = self.leftRotate(root.left)
            return self.rightRotate(root)

        elif balance < -1:
            if root.right.left == node:
                root.right = self.rightRotate(root.right)
            return self.leftRotate(root)

        self.updateMax(root)

        return root

    def leftRotate(self, x):

        y = x.right
        t = y.left

        y.left = x
        x.right = t

        x.height = 1 + max(self.getHeight(x.left), self.getHeight(x.right))
        y.height = 1 + max(self.getHeight(y.left), self.getHeight(y.right))

        self.updateMax(x)
        self.updateMax(y)

        return y

    def rightRotate(self, x):
        y = x.left
        t = y.right

        y.right = x
        x.left = t

        x.height = 1 + max(self.getHeight(x.left), self.getHeight(x.right))
        y.height = 1 + max(self.getHeight(y.left), self.getHeight(y.right))

        self.updateMax(x)
        self.updateMax(y)

        return y

    def getHeight(self, node):

        if not node:
            return 0
        return node.height

    def getBalance(self, node):

        if not node:
            return 0
        return self.getHeight(node.left) - self.getHeight(node.right)

    def query(self, node, begin, end):
        if node.left is not None and node.left.max >= begin:
            yield from self.query(node.left, begin, end)

        if node.inInterval(begin, end):
            yield node

        if node.right is not None and node.low <= end and node.right.max >= begin:
            yield from self.query(node.right, begin, end)

    def add(self, edge):
        start = edge.begin
        end = edge.end

        if (start, end) in self.nodes:
            node = self.nodes[(start, end)]
            node.edges.append(edge)
            self.number_of_edges += 1
            return

        node = Node(start, end)
        node.edges.append(edge)
        self.nodes[(start, end)] = node

        self.root = self.insert(self.root, node)
        self.number_of_edges += 1
        return

    def add_from(self, edges):
        for edge in edges:
            self.add(edge)
        return

    def slice(self, interval_start, interval_end, root=None):
        if root is None:
            root = self.root

        for node in self.query(root, interval_start, interval_end):
            yield from node.edges


class IntervalGraph:
    def __init__(self, **attr):
        self.node = {}
        self.adj = {}
        self.tree = IntervalTree()
        self.root = None
        self.attr = attr
        self.begin = math.inf
        self.end = -math.inf
        self.number_of_edges = 0

    def add_edge(self, u, v, begin, end):
        self.node.setdefault(u, {})
        self.node.setdefault(v, {})
        edge = Edge(u, v, begin, end)
        self.adj.setdefault(u, {}).setdefault(v, []).append(edge)
        self.adj.setdefault(v, {}).setdefault(u, []).append(edge)
        self.tree.add(Edge(u, v, begin, end))

        self.begin = min(self.begin, begin)
        self.end = max(self.end, end)
        self.number_of_edges += 1

    def interval(self):
        return self.begin, self.end

    def slice(self, interval_start, interval_end, root=None):
        if root is None:
            root = self.root

        yield from self.tree.slice(interval_start, interval_end, root)