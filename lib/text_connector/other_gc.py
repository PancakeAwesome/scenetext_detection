import numpy as np

class Graph(object):
    """docstring for Graph"""
    def __init__(self):
        self.graph = graph

    def sub_graph_connected(self):
        sub_graphs = []
        for index in range(self.graph.shape[0]):
            # 双方都是好邻居
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index
                sub_graphs.append([v])
                with self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)

        return sub_graphs