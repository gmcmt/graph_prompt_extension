import torch
import torch.nn as nn

class model_weight(nn.Module):
    def __init__(self):
        super(model_weight, self).__init__()
        
        
        # self.weight_1 = torch.nn.Parameter(torch.Tensor(1,1),requires_grad=True)
        # self.weight_2 = torch.nn.Parameter(torch.Tensor(1,1),requires_grad=True)
        # self.weight_3 = torch.nn.Parameter(torch.Tensor(1,1),requires_grad=True)
        # self.weight_4 = torch.nn.Parameter(torch.Tensor(1,1),requires_grad=True)
        self.temp = torch.nn.Parameter(torch.Tensor(4,1),requires_grad=True)

        
        self.reset_parameters()
        
    def reset_parameters(self):
        
        # torch.nn.init.uniform_(self.weight_1, a=0.0, b=5.0)
        # torch.nn.init.uniform_(self.weight_2, a=0.0, b=5.0)
        # torch.nn.init.uniform_(self.weight_3, a=0.0, b=5.0)
        # torch.nn.init.uniform_(self.weight_4, a=0.0, b=5.0)
        torch.nn.init.uniform_(self.temp, a=0.0, b=3)
    def forward(self, graph_adj,weight_id):
        # temp = torch.Tensor([self.weight_1,self.weight_2,self.weight_3,self.weight_4])
        temp = nn.functional.softmax(self.temp,dim =0)
        
        # size_ = graph_adj.size(0)
        # p = [i for i in range(size_)]
        # x = torch.tensor([p,p])
        # q = [self.weight for i in range(size_)]
        # tt = torch.sparse_coo_tensor(x,q,(size_,size_)).to(graph_adj.device)
        # graph_adj = (graph_adj + tt)
        graph_adj = graph_adj.to(self.temp.device)
        # if weight_id == 0:
        #     graph_adj = (graph_adj*(1.0+temp[0]))
        # elif weight_id == 1:
        #     graph_adj = (graph_adj*temp[1])
        # elif weight_id == 2:
        #     graph_adj = (graph_adj*temp[2])
        # else:
        #     graph_adj = (graph_adj*temp[3])

   
            
        return graph_adj
