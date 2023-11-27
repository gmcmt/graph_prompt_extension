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
from graph_prompt_layer import graph_prompt_layer_mean,graph_prompt_layer_linear_mean,graph_prompt_layer_linear_sum,\
    graph_prompt_layer_sum,graph_prompt_layer_feature_weighted_mean,graph_prompt_layer_feature_weighted_sum,node_prompt_layer_feature_weighted_sum

class GIN_P(torch.nn.Module):
    def __init__(self, config):
        super(GIN_P, self).__init__()

        # create networks
        # get_emb_dim 返回固定值：128,128(128为config值）
        # g_net为n层gcn网络，g_dim=hidden_dim
        self.act=torch.nn.ReLU()
        self.convs, self.bns, g_dim ,self.prompts = self.create_net(
            name="graph", input_dim=config["node_feature_dim"], hidden_dim=config["gcn_hidden_dim"],
            num_layers=config["gcn_graph_num_layers"], num_bases=config["gcn_num_bases"], regularizer=config["gcn_regularizer"],node_feature_dim = config["node_feature_dim"],)
        self.num_layers_num=config["gcn_graph_num_layers"]
        self.dropout=torch.nn.Dropout(p=config["dropout"])
                
        # create predict layersr
        # 这两个if语句在embedding网络的基础上增加了pattern和graph输入predict的维度数

    def create_net(self, name, input_dim, **kw):
        num_layers = kw.get("num_layers", 1)
        hidden_dim = kw.get("hidden_dim", 64)
        num_rels = kw.get("num_rels", 1)
        num_bases = kw.get("num_bases", 8)
        regularizer = kw.get("regularizer", "basis")
        dropout = kw.get("dropout", 0.5)
        feature_dim = kw.get("node_feature_dim",64)

        convs = torch.nn.ModuleList()
        bns = torch.nn.ModuleList()
        prompts = torch.nn.ModuleList()
        a = int((hidden_dim * num_layers)/3)
        prompt_1 = node_prompt_layer_feature_weighted_sum(feature_dim)
        prompt_2 = node_prompt_layer_feature_weighted_sum(a)
        prompt_3 = node_prompt_layer_feature_weighted_sum(a)
        prompt_4 = graph_prompt_layer_feature_weighted_sum(hidden_dim * num_layers)
        prompts.append(prompt_1)
        prompts.append(prompt_2)
        prompts.append(prompt_3)
        prompts.append(prompt_4)
        
        for i in range(num_layers):

            if i:
                nn = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), self.act, torch.nn.Linear(hidden_dim, hidden_dim))
            else:
                nn = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), self.act, torch.nn.Linear(hidden_dim, hidden_dim))
            conv = dgl.nn.pytorch.conv.GINConv(apply_func=nn,aggregator_type='sum')
            bn = torch.nn.BatchNorm1d(hidden_dim)

            convs.append(conv)
            bns.append(bn)

        return convs, bns, hidden_dim,prompts


    #def forward(self, pattern, pattern_len, graph, graph_len):
    def forward(self, graph, graph_len,prompt_id,scalar):
        graph_output = graph.ndata["feature"]
        xs = []
        if prompt_id == 0:
            graph_output = self.prompts[0](graph_output,graph_len)
            for i in range(self.num_layers_num):
                graph_output = F.relu(self.convs[i](graph,graph_output))
                graph_output = self.bns[i](graph_output)
                graph_output = self.dropout(graph_output)
                xs.append(graph_output)
        
            xpool= []
            for x in xs:
                graph_embedding = split_and_batchify_graph_feats(x, graph_len)[0]
                graph_embedding = torch.sum(graph_embedding, dim=1)
                xpool.append(graph_embedding)
            x = torch.cat(xpool, -1)
            #x is graph level embedding; xs is node level embedding
            embedding = torch.cat(xs, -1)
            embedding =split_and_batchify_graph_feats(embedding, graph_len)[0]
            embedding =embedding.mean(dim=1)
            return x,embedding
        elif prompt_id ==1:
            for i in range(self.num_layers_num):
                graph_output = F.relu(self.convs[i](graph,graph_output))
                graph_output = self.bns[i](graph_output)
                graph_output = self.dropout(graph_output)
                xs.append(graph_output)
                if i ==0:
                    graph_output = self.prompts[1](graph_output,graph.number_of_nodes())
                    
        
            xpool= []
            for x in xs:
                graph_embedding = split_and_batchify_graph_feats(x, graph_len)[0]
                graph_embedding = torch.sum(graph_embedding, dim=1)
                xpool.append(graph_embedding)
            x = torch.cat(xpool, -1)
            #x is graph level embedding; xs is node level embedding
            embedding = torch.cat(xs, -1)
            embedding =split_and_batchify_graph_feats(embedding, graph_len)[0]
            embedding =embedding.mean(dim=1)
            return x,embedding
        elif prompt_id ==2:
            for i in range(self.num_layers_num):
                graph_output = F.relu(self.convs[i](graph,graph_output))
                graph_output = self.bns[i](graph_output)
                graph_output = self.dropout(graph_output)
                xs.append(graph_output)
                if i ==1:
                    graph_output = self.prompts[2](graph_output,graph.number_of_nodes())
                    
        
            xpool= []
            for x in xs:
                graph_embedding = split_and_batchify_graph_feats(x, graph_len)[0]
                graph_embedding = torch.sum(graph_embedding, dim=1)
                xpool.append(graph_embedding)
            x = torch.cat(xpool, -1)
            embedding = torch.cat(xs, -1)
            embedding =split_and_batchify_graph_feats(embedding, graph_len)[0]
            embedding =embedding.mean(dim=1)
            #x is graph level embedding; xs is node level embedding
            return x,embedding
        elif prompt_id == 3:
            for i in range(self.num_layers_num):
                graph_output = F.relu(self.convs[i](graph,graph_output))
                graph_output = self.bns[i](graph_output)
                graph_output = self.dropout(graph_output)
                xs.append(graph_output)
                   
            xpool= []
            for x in xs:
                graph_embedding = split_and_batchify_graph_feats(x, graph_len)[0]
                graph_embedding = torch.sum(graph_embedding, dim=1)
                xpool.append(graph_embedding)
            x = torch.cat(xpool, -1)
            embedding = torch.cat(xs, -1)
            embedding = self.prompts[3](embedding, graph_len)*scalar
            
            #x is graph level embedding; xs is node level embedding
            return x,embedding
        else:
            for i in range(self.num_layers_num):
                graph_output = F.relu(self.convs[i](graph,graph_output))
                graph_output = self.bns[i](graph_output)
                graph_output = self.dropout(graph_output)
                xs.append(graph_output)
              
                    
        
            xpool= []
            for x in xs:
                graph_embedding = split_and_batchify_graph_feats(x, graph_len)[0]
                graph_embedding = torch.sum(graph_embedding, dim=1)
                xpool.append(graph_embedding)
            x = torch.cat(xpool, -1)
            embedding = torch.cat(xs, -1)            
            #x is graph level embedding; xs is node level embedding
            return x,embedding
            
            
