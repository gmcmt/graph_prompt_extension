import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import copy
from functools import partial
from dgl.nn.pytorch.conv import RelGraphConv
from basemodel import GraphAdjModel
import math
from utils import map_activation_str_to_layer, split_and_batchify_graph_feats,GetAdj

class node_finetuning_layer(nn.Module):
    def __init__(self, input_dim,output_dim):
        super(node_finetuning_layer, self).__init__()
        self.linear=torch.nn.Linear(input_dim,output_dim)
        #self.dropout=torch.nn.Dropout(0.2)


    def forward(self,graph_embedding, graph_len):
        #not the follows problem
        graph_embedding=self.linear(graph_embedding)
        #graph_embedding=torch.nn.functional.normalize(graph_embedding,dim=1)
        #graph_embedding=F.leaky_relu(graph_embedding,0.2)
        result = F.log_softmax(graph_embedding, dim=1)
        return result


'''class graph_finetuning_layer(nn.Module):
    def __init__(self, input_dim,output_dim):
        super(graph_finetuning_layer, self).__init__()
        self.linear=torch.nn.Linear(input_dim,output_dim)
        self.softmax=torch.nn.Softmax(dim=1)

    def forward(self,graph_embedding, graph_len):
        graph_embedding=split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        graph_embedding=torch.sum(graph_embedding,dim=1)
        graph_embedding=self.linear(graph_embedding)
        graph_embedding=F.leaky_relu(graph_embedding)
        graph_embedding=F.log_softmax(graph_embedding,dim=1)
        #graph_embedding=F.softmax(graph_embedding,dim=1)
        #graph_embedding=torch.argmax(graph_embedding,dim=1,keepdim=True).float()
        #result=self.softmax(F.leaky_relu(graph_embedding))
        #index=result.permute(1,0)[0]
        #index=index.unsqueeze(dim=1)
        index=torch.argmax(graph_embedding,dim=1,keepdim=True).float()
        index.requires_grad_(True)
        print(index.requires_grad)
        #return result
        #return graph_embedding
        return index'''