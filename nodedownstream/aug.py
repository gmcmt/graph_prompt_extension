import torch
import copy
import random
import pdb
import scipy.sparse as sp
import numpy as np

def main():
    pass


def aug_random_mask(input_feature, drop_percent=0.2):
    
    node_num = input_feature.shape[1]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    aug_feature = copy.deepcopy(input_feature)
    zeros = torch.zeros_like(aug_feature[0][0])
    for j in mask_idx:
        aug_feature[0][j] = zeros
    return aug_feature


def aug_random_edge(input_adj, drop_percent=0.2):

    percent = drop_percent / 2
    row_idx, col_idx = input_adj.nonzero()

    index_list = []
    for i in range(len(row_idx)):
        index_list.append((row_idx[i], col_idx[i]))

    single_index_list = []
    for i in list(index_list):
        single_index_list.append(i)
        index_list.remove((i[1], i[0]))
    
    
    edge_num = int(len(row_idx) / 2)      # 9228 / 2
    add_drop_num = int(edge_num * percent / 2) 
    aug_adj = copy.deepcopy(input_adj.todense().tolist())

    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)

    
    for i in drop_idx:
        aug_adj[single_index_list[i][0]][single_index_list[i][1]] = 0
        aug_adj[single_index_list[i][1]][single_index_list[i][0]] = 0
    
    '''
    above finish drop edges
    '''
    node_num = input_adj.shape[0]
    l = [(i, j) for i in range(node_num) for j in range(i)]
    add_list = random.sample(l, add_drop_num)

    for i in add_list:
        
        aug_adj[i[0]][i[1]] = 1
        aug_adj[i[1]][i[0]] = 1
    
    aug_adj = np.matrix(aug_adj)
    aug_adj = sp.csr_matrix(aug_adj)
    return aug_adj


def aug_drop_node(input_fea, input_adj, drop_percent=0.2):

    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)

    node_num = input_fea.shape[0]
    drop_num = int(node_num * drop_percent)    # number of drop nodes
    all_node_list = [i for i in range(node_num)]

    drop_node_list = sorted(random.sample(all_node_list, drop_num))

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj

def aug_subgraph_CL(graph, drop_percent=0.2):
        # input_adj = graph.adjacency_matrix().to_dense()
    # input_fea = input_fea.squeeze(0)
    edge_num = graph.batch_num_edges().tolist()
    # all_edge_list = [i for i in range(edge_num)]
    # s_node_num = int(edge_num * (1 - drop_percent))
    # center_node_id = random.randint(0, node_num - 1)
    # sub_node_id_list = [center_node_id]
    # all_neighbor_list = []
    # for i in range(s_node_num - 1):
        
    #     all_neighbor_list += torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()
    #     # print(torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False))
    #     all_neighbor_list = list(set(all_neighbor_list))
    #     new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]
    #     if len(new_neighbor_list) != 0:
    #         new_node = random.sample(new_neighbor_list, 1)[0]
    #         sub_node_id_list.append(new_node)
    #     else:
    #         break

    
    # print("hhhhhhh")
    # print(drop_node_list)
    # a = graph_len.squeeze(1).tolist()
    sub_edge_id_list = []
    tag = 0
    for i in range(len(edge_num)):
        s_edge_num = int(edge_num[i] * drop_percent)
        temp = random.sample(range(0,edge_num[i]),s_edge_num)
        sub_edge_id_list += [(x+tag) for x in temp]
        tag+=edge_num[i]
        # edge_num[i] = edge_num[i]-s_node_num
    
    drop_edge_list = sub_edge_id_list
 
    # a = torch.IntTensor(a).unsqueeze(1)
    
    
    
    graph.remove_edges(drop_edge_list)
    
    # return graph.subgraph(sub_node_id_list),a
    return graph

    return graph,graph_len
def aug_subgraph(graph,graph_len, drop_percent=0.2):
    
    # input_adj = graph.adjacency_matrix().to_dense()
    # input_fea = input_fea.squeeze(0)
    node_num = graph.ndata['feature'].shape[0]
    # all_node_list = [i for i in range(node_num)]
    # s_node_num = int(node_num * (1 - drop_percent))
    # center_node_id = random.randint(0, node_num - 1)
    # sub_node_id_list = [center_node_id]
    # all_neighbor_list = []
    # for i in range(s_node_num - 1):
        
    #     all_neighbor_list += torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()
    #     # print(torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False))
    #     all_neighbor_list = list(set(all_neighbor_list))
    #     new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]
    #     if len(new_neighbor_list) != 0:
    #         new_node = random.sample(new_neighbor_list, 1)[0]
    #         sub_node_id_list.append(new_node)
    #     else:
    #         break

    
    # print("hhhhhhh")
    # print(drop_node_list)
    a = graph_len.squeeze(1).tolist()
    sub_node_id_list = []
    tag = 0
    for i in range(len(a)):
        s_node_num = int(a[i] * drop_percent)
        
 
        temp = random.sample(range(0,a[i]),s_node_num)
        sub_node_id_list += [(x+tag) for x in temp]
        tag+=a[i]
        a[i] = a[i]-s_node_num
    
    drop_node_list = sub_node_id_list
 
    a = torch.IntTensor(a).unsqueeze(1)
    
    

    graph.remove_edges(drop_node_list)
    
    # return graph.subgraph(sub_node_id_list),a
    return graph,a

#用于Flickr数据集在DGI方法下删除结点

def aug_subgraph_F(graph, drop_percent=0.2):
    
    node_num = graph.num_nodes()  
    # print("hhhhhhh")
    # print(drop_node_list)



    s_node_num = int(node_num * drop_percent)
    

    drop_node_list = random.sample(range(0,node_num),s_node_num)
    
   
 
    node_num = node_num - s_node_num
    
    try:
        graph.remove_nodes(drop_node_list)
    except:
        pass
    
    # return graph.subgraph(sub_node_id_list),a
    return graph,node_num





def delete_row_col(input_matrix, drop_list, only_row=False):

    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out



    



    

     

    







if __name__ == "__main__":
    main()
    
