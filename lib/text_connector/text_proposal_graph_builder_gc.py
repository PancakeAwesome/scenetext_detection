from .text_connect_cfg import Config as TextLineCfg
from .other import Graph
import numpy as np

class TextProposalGraphBuilder():
    """Build Text proposals into a graph.
    基于图的文本构造方法
    """
    def get_successions(self, index):
            box=self.text_proposals[index]
            results=[]
            for left in range(int(box[0])+1, min(int(box[0])+TextLineCfg.MAX_HORIZONTAL_GAP+1, self.im_size[1])):
                adj_box_indices=self.boxes_table[left]
                for adj_box_index in adj_box_indices:
                    if self.meet_v_iou(adj_box_index, index):
                        results.append(adj_box_index)
                if len(results)!=0:
                    return results
            return results

    def get_precursors(self, index):
        box=self.text_proposals[index]
        results=[]
        for left in range(int(box[0])-1, max(int(box[0]-TextLineCfg.MAX_HORIZONTAL_GAP), 0)-1, -1):
            adj_box_indices=self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results)!=0:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        precursors=self.get_precursors(succession_index)
        if self.scores[index]>=np.max(self.scores[precursors]):
            return True
        return False

    def meet_v_iou(self, index1, index2):
        def overlaps_v(index1, index2):
            h1=self.heights[index1]
            h2=self.heights[index2]
            y0=max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1=min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1-y0+1)/min(h1, h2)

        def size_similarity(index1, index2):
            h1=self.heights[index1]
            h2=self.heights[index2]
            return min(h1, h2)/max(h1, h2)

        return overlaps_v(index1, index2)>=TextLineCfg.MIN_V_OVERLAPS and \
               size_similarity(index1, index2)>=TextLineCfg.MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1

        # 构造空的列表框，列表的长度为图片的宽度W像素
        boxes_table = [[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table = boxes_table

        # 构造proposals_numXproposals_num的graph
        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            # 得到和index的box的好邻居
            # * 文本线构造算法（多个细长的proposal合并成一条文本线）（边缘细化）
            #     * 主要思想：每两个相近的proposal组成一个pair，合并不同的pair直到无法再合并为止（没有公共元素）
            #     * 判断两个proposal，Bi和Bj组成pair的条件：
            # 1. Bj->Bi， 且Bi->Bj。（Bj->Bi表示Bj是Bi的最好邻居）
            # 2. Bj->Bi条件1：Bj是Bi的邻居中距离Bi最近的，且该距离小于50个像素
            # 3. Bj->Bi条件2：Bj和Bi的vertical overlap大于0.7
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            # 取得分最高的好邻居
            successions_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                graph[index, successions_index] = True
        return Graph(graph)

        
