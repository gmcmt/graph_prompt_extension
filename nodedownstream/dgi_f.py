import torch
import torch.nn as nn
from layers import AvgReadout, Discriminator, Discriminator2
import pdb
from gin_flickr_DGI import GIN
from utils import split_and_batchify_graph_feats
class DGI(nn.Module):
    def __init__(self, n_in, n_h,config):
        super(DGI, self).__init__()
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h*3)
        # self.disc2 = Discriminator2(n_h)
        self.gin = GIN(config)
        self.config = config


    def forward(self, graph,graph_shuf, graph_1, graph_2, graph_len, graph_len_1, graph_len_2, sparse, msk, samp_bias1, samp_bias2, aug_type):

        c_0,h_0 = self.gin(graph, graph_len)

        if aug_type == 'edge':

            # h_1 = self.gcn(seq1, aug_adj1, sparse)
            # h_3 = self.gcn(seq1, aug_adj2, sparse)
            pass

        elif aug_type == 'mask':

            # h_1 = self.gcn(seq3, adj, sparse)
            # h_3 = self.gcn(seq4, adj, sparse)
            pass

        elif aug_type == 'node' or aug_type == 'subgraph':

            c_1,h_1 = self.gin(graph_1, graph_len_1)
            c_3,h_3 = self.gin(graph_2, graph_len_2)
            
        else:
            assert False
            
        
        c_2,h_2 = self.gin(graph_shuf,graph_len)
        h_0,h_2 = self.sigm(h_0),self.sigm(h_2)
        
        
        # len_1 = int(h_0.shape[1])
        # len_2 = int(h_2.shape[1])
        # ret1 = self.disc(c_1, h_0, h_2, samp_bias1, samp_bias2)
        # ret2 = self.disc(c_3, h_0, h_2, samp_bias1, samp_bias2)
        device = self.config["gpu_id"]
        # ret = ret1 + ret2
        # return ret,int(h_0.shape[1]),int(h_2.shape[1])
        graph.add_self_loop()
        graph_1.add_self_loop()
        graph_2.add_self_loop()
        adj = graph.adjacency_matrix()
        adj_1 = graph_1.adjacency_matrix()
        adj_2 = graph_2.adjacency_matrix()
        adj = adj.to(device)
        adj_1 = adj_1.to(device)
        adj_2 = adj_2.to(device)
        # for count in range(self.config["pretrain_hop_num"]):
        #     h_0 = torch.matmul(adj, h_0)
        #     # h_1 = torch.matmul(adj_1, h_1)
        #     h_2 = torch.matmul(adj, h_2)
        #     # h_3 = torch.matmul(adj_2, h_3)
        return h_0,c_1,h_2,c_3

    # Detach the return variables
    def embed(self, graph, graph_len):
        return self.gin(graph,graph_len)
        

        

