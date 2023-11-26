import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import copy
from functools import partial
from dgl.nn.pytorch.conv import RelGraphConv
from basemodel import GraphAdjModel
from utils import map_activation_str_to_layer, split_and_batchify_graph_feats,GetAdj


class GCN(torch.nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()

        # create networks
        # get_emb_dim 返回固定值：128,128(128为config值）
        # g_net为n层gcn网络，g_dim=hidden_dim
        self.act=torch.nn.ReLU()
        self.g_net, g_dim = self.create_net(
            name="graph", input_dim=config["node_feature_dim"], hidden_dim=config["gcn_hidden_dim"],
            num_layers=config["gcn_graph_num_layers"], num_bases=config["gcn_num_bases"], regularizer=config["gcn_regularizer"])
        self.num_layers_num=config["gcn_graph_num_layers"]

        # create predict layersr
        # 这两个if语句在embedding网络的基础上增加了pattern和graph输入predict的维度数

    def create_net(self, name, input_dim, **kw):
        num_layers = kw.get("num_layers", 1)
        hidden_dim = kw.get("hidden_dim", 64)
        num_rels = kw.get("num_rels", 1)
        num_bases = kw.get("num_bases", 8)
        regularizer = kw.get("regularizer", "basis")
        dropout = kw.get("dropout", 0.5)


        self.convs = torch.nn.ModuleList()

        for i in range(num_layers):

            if i:
                conv = dgl.nn.pytorch.conv.GraphConv(in_feats=hidden_dim, out_feats=hidden_dim,allow_zero_in_degree=True)
            else:
                conv = dgl.nn.pytorch.conv.GraphConv(in_feats=input_dim, out_feats=hidden_dim,allow_zero_in_degree=True)

            self.convs.append(conv)

        return self.convs, hidden_dim


    #def forward(self, pattern, pattern_len, graph, graph_len):
    def forward(self, graph, graph_len):
        #bsz = pattern_len.size(0)
        # filter_gate选出了graph中与同构无关的节点的mask
        #gate = self.get_filter_gate(pattern, pattern_len, graph, graph_len)
        graph_output = graph.ndata["feature"]
        xs = []
        for i in range(self.num_layers_num):
            graph_output = F.relu(self.convs[i](graph,graph_output))
            xs.append(graph_output)
        xpool= []
        for x in xs:
            graph_embedding = x
            graph_embedding = torch.sum(graph_embedding, dim=1)
            xpool.append(graph_embedding)
        x = torch.cat(xpool, -1)
        return x,torch.cat(xs, -1)
