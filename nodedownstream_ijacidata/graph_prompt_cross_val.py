import torch
import os
import numpy as np
import dgl
import logging
import datetime
import math
import sys
import gc
import json
import time
import torch.nn.functional as F
import warnings
import itertools
from functools import partial
from collections import OrderedDict
from torch.utils.data import DataLoader
from dgllife.utils import ConsecutiveSplitter


try:
    from torch.utils.tensorboard import SummaryWriter
except BaseException as e:
    from tensorboardX import SummaryWriter
from dataset import Sampler, EdgeSeqDataset, GraphAdjDataset, GraphAdjDataset_DGL_Input
from utils import anneal_fn, get_enc_len, pretrain_load_data, get_linear_schedule_with_warmup,\
    bp_compute_abmae,compareloss,label2onehot,correctness_GPU,correctness,macrof1,weightf1,distance2center,center_embedding,distance2center2
'''from cnn import CNN
from rnn import RNN
from txl import TXL
from rgcn import RGCN
from rgin import RGIN
from gat import GAT
from gcn import GCN
from graphsage import GraphSage
from gin import GIN'''
from gat import GAT
from gcn_onehot import GCN
from gin_onehot import GIN
from graphsage import Graphsage
from graph_finetuning_layer import graph_finetuning_layer
from graph_prompt_layer import graph_prompt_layer_mean,graph_prompt_layer_linear_mean,graph_prompt_layer_linear_sum,\
    graph_prompt_layer_sum,graph_prompt_layer_feature_weighted_mean,graph_prompt_layer_feature_weighted_sum


warnings.filterwarnings("ignore")
INF = float("inf")

train_config = {
    "max_npv": 8,  # max_number_pattern_vertices: 8, 16, 32
    "max_npe": 8,  # max_number_pattern_edges: 8, 16, 32
    "max_npvl": 8,  # max_number_pattern_vertex_labels: 8, 16, 32
    "max_npel": 8,  # max_number_pattern_edge_labels: 8, 16, 32

    "max_ngv": 64,  # max_number_graph_vertices: 64, 512,4096
    "max_nge": 256,  # max_number_graph_edges: 256, 2048, 16384
    "max_ngvl": 16,  # max_number_graph_vertex_labels: 16, 64, 256
    "max_ngel": 16,  # max_number_graph_edge_labels: 16, 64, 256

    "base": 2,

    "gpu_id": -1,
    "num_workers": 12,

    "epochs": 200,
    "batch_size": 512,
    "update_every": 1,  # actual batch_sizer = batch_size * update_every
    "print_every": 100,
    "init_emb": "Equivariant",  # None, Orthogonal, Normal, Equivariant
    "share_emb": True,  # sharing embedding requires the same vector length
    "share_arch": True,  # sharing architectures
    "dropout": 0.2,
    "dropatt": 0.2,

    "reg_loss": "MSE",  # MAE, MSEl
    "bp_loss": "MSE",  # MAE, MSE
    "bp_loss_slp": "anneal_cosine$1.0$0.01",  # 0, 0.01, logistic$1.0$0.01, linear$1.0$0.01, cosine$1.0$0.01,
    # cyclical_logistic$1.0$0.01, cyclical_linear$1.0$0.01, cyclical_cosine$1.0$0.01
    # anneal_logistic$1.0$0.01, anneal_linear$1.0$0.01, anneal_cosine$1.0$0.01
    "lr": 0.001,
    "weight_decay": 0.00001,
    "max_grad_norm": 8,

    "pretrain_model": "GCN",

    "emb_dim": 128,
    "activation_function": "leaky_relu",  # sigmoid, softmax, tanh, relu, leaky_relu, prelu, gelu

    "filter_net": "MaxGatedFilterNet",  # None, MaxGatedFilterNet
    "predict_net": "SumPredictNet",  # MeanPredictNet, SumPredictNet, MaxPredictNet,
    "predict_net_add_enc": True,
    "predict_net_add_degree": True,

    # MeanAttnPredictNet, SumAttnPredictNet, MaxAttnPredictNet,
    # MeanMemAttnPredictNet, SumMemAttnPredictNet, MaxMemAttnPredictNet,
    # DIAMNet
    # "predict_net_add_enc": True,
    # "predict_net_add_degree": True,
    "txl_graph_num_layers": 3,
    "txl_pattern_num_layers": 3,
    "txl_d_model": 128,
    "txl_d_inner": 128,
    "txl_n_head": 4,
    "txl_d_head": 4,
    "txl_pre_lnorm": True,
    "txl_tgt_len": 64,
    "txl_ext_len": 0,  # useless in current settings
    "txl_mem_len": 64,
    "txl_clamp_len": -1,  # max positional embedding index
    "txl_attn_type": 0,  # 0 for Dai et al, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.
    "txl_same_len": False,

    "gcn_num_bases": 8,
    "gcn_regularizer": "bdd",  # basis, bdd
    "gcn_graph_num_layers": 3,
    "gcn_hidden_dim": 128,
    "gcn_ignore_norm": False,  # ignorm=True -> RGCN-SUM

    "graph_dir": "../data/debug/graphs",
    "save_data_dir": "../data/debug",
    "save_model_dir": "../dumps/debug",
    "save_pretrain_model_dir": "../dumps/MUTAGPreTrain/GCN",
    "graphslabel_dir":"../data/debug/graphs",
    "downstream_graph_dir": "../data/debug/graphs",
    "downstream_save_data_dir": "../data/debug",
    "downstream_save_model_dir": "../dumps/debug",
    "downstream_graphslabel_dir":"../data/debug/graphs",
    "temperature": 0.01,
    "graph_finetuning_input_dim": 8,
    "graph_finetuning_output_dim": 2,
    "graph_label_num":2,
    "seed": 0,
    "update_pretrain": False,
    "dropout": 0.5,
    "prompt": "SUM",
    "prompt_output_dim": 2,
}


def train(pretrain_model, model, optimizer, scheduler, data_type, data_loader, device, config, epoch, logger=None, writer=None):
    epoch_step = len(data_loader)
    total_step = config["epochs"] * epoch_step
    total_reg_loss = 0
    total_bp_loss = 0
    batchcnt=0
    total_acc=0
    total_cnt = 1e-6

    if config["reg_loss"] == "MAE":
        reg_crit = lambda pred, target: F.l1_loss(F.relu(pred), target)
    elif config["reg_loss"] == "MSE":
        reg_crit = lambda pred, target: F.mse_loss(F.relu(pred), target)
    elif config["reg_loss"] == "SMSE":
        reg_crit = lambda pred, target: F.smooth_l1_loss(F.relu(pred), target)
    elif config["reg_loss"] == "NLL":
        reg_crit = lambda pred, target: F.nll_loss(F.relu(pred), target)
    elif config["reg_loss"] == "ABMAE":
        reg_crit = lambda pred, target: bp_compute_abmae(F.leaky_relu(pred), target) + 0.8 * F.l1_loss(F.relu(pred),
                                                                                                       target)
    else:
        raise NotImplementedError

    if config["bp_loss"] == "MAE":
        bp_crit = lambda pred, target, neg_slp: F.l1_loss(F.leaky_relu(pred, neg_slp), target)
    elif config["bp_loss"] == "MSE":
        bp_crit = lambda pred, target, neg_slp: F.mse_loss(F.leaky_relu(pred, neg_slp), target)
    elif config["bp_loss"] == "SMSE":
        bp_crit = lambda pred, target, neg_slp: F.smooth_l1_loss(F.leaky_relu(pred, neg_slp), target)
    elif config["bp_loss"] == "NLL":
        bp_crit = lambda pred, target,neg_slp: F.nll_loss(pred, target)
    elif config["bp_loss"] == "ABMAE":
        bp_crit = lambda pred, target, neg_slp: bp_compute_abmae(F.leaky_relu(pred, neg_slp), target) + 0.8 * F.l1_loss(
            F.leaky_relu(pred, neg_slp), target, reduce="none")
    else:
        raise NotImplementedError

    model.train()
    total_time = 0
    l2onehot=label2onehot(train_config["graph_label_num"],device)
    label_num=torch.tensor(train_config["graph_label_num"]).to(device)

    acc=0
    for batch_id, batch in enumerate(data_loader):
        ids, graph_label, graph, graph_len = batch
        # print(batch)
        cnt = graph_label.shape[0]
        total_cnt+=cnt
        graph = graph.to(device)
        graph_label=graph_label.to(device)
        graph_len = graph_len.to(device)
        batchcnt+=1
        s = time.time()

        x,embedding=pretrain_model(graph,graph_len)

        graph_label_onehot=l2onehot(graph_label)
        #for mutag: embedding size is (92,128)
        #test if pretrain_model result in grad disappear
        #proves the bug isnt because of this
        #embedding=torch.ones(92,128)
        #embedding.requires_grad_(True)
        embedding = model(embedding, graph_len)
        c_embedding=center_embedding(embedding,graph_label,label_num)
        distance=distance2center(embedding,c_embedding)
        #print(distance)
        distance = -1*F.normalize(distance, dim=1)
        #distance=distance2center2(embedding,c_embedding)

        #print('distance: ',distance )
        pred=F.log_softmax(distance,dim=1)

        reg_loss = reg_crit(pred, graph_label.squeeze().type(torch.LongTensor).to(device))

        #reg_loss = reg_crit(pred, graph_label_onehot)
        reg_loss.requires_grad_(True)

        if isinstance(config["bp_loss_slp"], (int, float)):
            neg_slp = float(config["bp_loss_slp"])
        else:
            bp_loss_slp, l0, l1 = config["bp_loss_slp"].rsplit("$", 3)
            neg_slp = anneal_fn(bp_loss_slp, batch_id + epoch * epoch_step, T=total_step // 4, lambda0=float(l0),
                                lambda1=float(l1))
        bp_loss = bp_crit(pred.float(), graph_label.squeeze().type(torch.LongTensor).to(device),neg_slp)

        #bp_loss = bp_crit(pred.float(), graph_label_onehot.float(), neg_slp)
        bp_loss.requires_grad_(True)
        _pred = torch.argmax(pred, dim=1, keepdim=True)
        acc = correctness_GPU(_pred, graph_label)

        # float
        reg_loss_item = reg_loss.item()
        bp_loss_item = bp_loss.item()
        total_reg_loss += reg_loss_item
        total_bp_loss += bp_loss_item
        total_acc+=acc
        if writer:
            writer.add_scalar("%s/REG-%s" % (data_type, config["reg_loss"]), reg_loss_item,
                              epoch * epoch_step + batch_id)
            writer.add_scalar("%s/BP-%s" % (data_type, config["bp_loss"]), bp_loss_item, epoch * epoch_step + batch_id)

        if logger and (batch_id % config["print_every"] == 0 or batch_id == epoch_step - 1):
            logger.info(
                "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}\treg loss: {:0>10.6f}\tbp loss: {:0>16.6f}\tacc: {:0>10.6f}".format(
                    epoch, config["epochs"], data_type, batch_id, epoch_step,
                    reg_loss_item, bp_loss_item, acc))

        bp_loss.backward()
        '''for name, parms in model.named_parameters():
            print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                  ' -->grad_value:', parms.grad)'''

        if (config["update_every"] < 2 or batch_id % config["update_every"] == 0 or batch_id == epoch_step - 1):
            if config["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            if scheduler is not None:
                scheduler.step(epoch * epoch_step + batch_id)
            optimizer.step()
            optimizer.zero_grad()
        e=time.time()
        total_time+=e-s

    mean_reg_loss = total_reg_loss / total_cnt
    mean_bp_loss = total_bp_loss / total_cnt
    mean_acc=total_acc/batchcnt
    if writer:
        writer.add_scalar("%s/REG-%s-epoch" % (data_type, config["reg_loss"]), mean_reg_loss, epoch)
        writer.add_scalar("%s/BP-%s-epoch" % (data_type, config["bp_loss"]), mean_bp_loss, epoch)
    if logger:
        logger.info("epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\treg loss: {:0>10.6f}\tbp loss: {:0>16.6f}\tacc: {:0>1.6f}".format(
            epoch, config["epochs"], data_type, mean_reg_loss, mean_bp_loss,mean_acc))

    gc.collect()
    return mean_reg_loss, mean_bp_loss, total_time, mean_acc, c_embedding


def evaluate(pretrain_model, model, data_type, data_loader, device, config, epoch, c_embedding, logger=None, writer=None):
    epoch_step = len(data_loader)
    total_reg_loss = 0
    total_step = config["epochs"] * epoch_step
    total_bp_loss = 0
    total_acc=0
    batchcnt=0
    total_cnt = 1e-6

    evaluate_results = {"data": {"id": list(), "counts": list(), "pred": list()},
                        "error": {"mae": INF, "mse": INF},
                        "time": {"avg": list(), "total": 0.0}}

    if config["reg_loss"] == "MAE":
        reg_crit = lambda pred, target: F.l1_loss(F.relu(pred), target, reduce="none")
    elif config["reg_loss"] == "MSE":
        reg_crit = lambda pred, target: F.mse_loss(F.relu(pred), target, reduce="none")
    elif config["reg_loss"] == "SMSE":
        reg_crit = lambda pred, target: F.smooth_l1_loss(F.relu(pred), target, reduce="none")
    elif config["reg_loss"] == "NLL":
        reg_crit = lambda pred, target: F.nll_loss(F.relu(pred), target)
    elif config["reg_loss"] == "ABMAE":
        reg_crit = lambda pred, target: bp_compute_abmae(F.relu(pred), target) + 0.8 * F.l1_loss(F.relu(pred), target,
                                                                                                 reduce="none")
    else:
        raise NotImplementedError

    if config["bp_loss"] == "MAE":
        bp_crit = lambda pred, target, neg_slp: F.l1_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    elif config["bp_loss"] == "MSE":
        bp_crit = lambda pred, target, neg_slp: F.mse_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    elif config["bp_loss"] == "NLL":
        bp_crit = lambda pred, target,neg_slp: F.nll_loss(pred, target)
    elif config["bp_loss"] == "SMSE":
        bp_crit = lambda pred, target, neg_slp: F.smooth_l1_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    elif config["bp_loss"] == "ABMAE":
        bp_crit = lambda pred, target, neg_slp: bp_compute_abmae(F.leaky_relu(pred, neg_slp), target) + 0.8 * F.l1_loss(
            F.leaky_relu(pred, neg_slp), target, reduce="none")
    else:
        raise NotImplementedError
    l2onehot=label2onehot(train_config["graph_label_num"],device)

    model.eval()
    #l2onehot=label2onehot(train_config["graph_label_num"])
    total_time = 0
    with torch.no_grad():
        for batch_id, batch in enumerate(data_loader):
            ids, graph_label, graph, graph_len = batch
            # print(batch)
            cnt = graph_label.shape[0]
            total_cnt+=cnt
            batchcnt+=1
            graph = graph.to(device)
            graph_label = graph_label.to(device)
            graph_len = graph_len.to(device)
            graph_label_onehot = l2onehot(graph_label)

            s = time.time()
            x,embedding = pretrain_model(graph, graph_len)
            embedding = model(embedding, graph_len)
            distance = distance2center(embedding, c_embedding)
            # print(distance)
            distance = -1 * F.normalize(distance, dim=1)
            # distance=distance2center2(embedding,c_embedding)

            # print('distance: ',distance )
            pred = F.log_softmax(distance, dim=1)
            _pred = torch.argmax(pred, dim=1, keepdim=True)

            #reg_loss = reg_crit(pred, graph_label_onehot)
            reg_loss = reg_crit(pred, graph_label.squeeze().type(torch.LongTensor).to(device))

            if isinstance(config["bp_loss_slp"], (int, float)):
                neg_slp = float(config["bp_loss_slp"])
            else:
                bp_loss_slp, l0, l1 = config["bp_loss_slp"].rsplit("$", 3)
                neg_slp = anneal_fn(bp_loss_slp, batch_id + epoch * epoch_step, T=total_step // 4, lambda0=float(l0),
                                    lambda1=float(l1))
            #bp_loss = bp_crit(pred.float(), graph_label_onehot.float(), neg_slp)
            bp_loss = bp_crit(pred, graph_label.squeeze().type(torch.LongTensor).to(device), neg_slp)

            #graph_label_onehot=l2onehot(graph_label)
            _pred=torch.argmax(pred,dim=1,keepdim=True)
            # float
            reg_loss_item = reg_loss.item()
            bp_loss_item = bp_loss.item()
            total_reg_loss += reg_loss_item * cnt
            total_bp_loss += bp_loss_item * cnt
            evaluate_results["data"]["id"].extend(ids)
            evaluate_results["data"]["counts"].extend(graph_label.view(-1).tolist())
            #evaluate_results["time"]["total"] += (et-st)
            #avg_t = (et-st) / (cnt + 1e-8)
            #evaluate_results["time"]["avg"].extend([avg_t]*cnt)
            evaluate_results["data"]["pred"].extend(_pred.cpu().view(-1).tolist())

            evaluate_results["error"]["mae"] += F.l1_loss(F.leaky_relu(pred.float(),neg_slp), graph_label_onehot.float(), reduce="none").sum().item()
            evaluate_results["error"]["mse"] += F.mse_loss(F.leaky_relu(pred.float(),neg_slp), graph_label_onehot.float(), reduce="none").sum().item()
            accuracy = correctness(evaluate_results["data"]["pred"], evaluate_results["data"]["counts"])
            total_acc+=accuracy
            if writer:
                writer.add_scalar("%s/REG-%s" % (data_type, config["reg_loss"]), reg_loss_item,
                                  epoch * epoch_step + batch_id)
                writer.add_scalar("%s/BP-%s" % (data_type, config["bp_loss"]), bp_loss_item,
                                  epoch * epoch_step + batch_id)

            if logger and batch_id == epoch_step - 1:
                logger.info(
                    "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}\treg loss: {:0>10.6f}\tbp loss: {:0>16.6f}\tacc: {:0>10.6f}".format(
                        epoch, config["epochs"], data_type, batch_id, epoch_step,
                        reg_loss_item, bp_loss_item, accuracy))
        mean_reg_loss = total_reg_loss / total_cnt
        mean_bp_loss = total_bp_loss / total_cnt
        mean_acc=total_acc/batchcnt
        if writer:
            writer.add_scalar("%s/REG-%s-epoch" % (data_type, config["reg_loss"]), mean_reg_loss, epoch)
            writer.add_scalar("%s/BP-%s-epoch" % (data_type, config["bp_loss"]), mean_bp_loss, epoch)
        if logger:
            logger.info("epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\treg loss: {:0>10.6f}\tbp loss: {:0>16.6f}\tacc:{:0>1.6f}".format(
                epoch, config["epochs"], data_type, mean_reg_loss, mean_bp_loss,mean_acc))

        evaluate_results["error"]["mae"] = evaluate_results["error"]["mae"] / total_cnt
        evaluate_results["error"]["mse"] = evaluate_results["error"]["mse"] / total_cnt

        macrof=macrof1(evaluate_results["data"]["pred"],evaluate_results["data"]["counts"])
        weightedf=weightf1(evaluate_results["data"]["pred"],evaluate_results["data"]["counts"])

    gc.collect()
    return mean_reg_loss, mean_bp_loss, evaluate_results, mean_acc, accuracy,macrof,weightedf


if __name__ == "__main__":
    for i in range(1, len(sys.argv), 2):
        arg = sys.argv[i]
        value = sys.argv[i + 1]

        if arg.startswith("--"):
            arg = arg[2:]
        if arg not in train_config:
            print("Warning: %s is not surported now." % (arg))
            continue
        train_config[arg] = value
        try:
            value = eval(value)
            if isinstance(value, (int, float)):
                train_config[arg] = value
        except:
            pass

    torch.manual_seed(train_config["seed"])
    np.random.seed(train_config["seed"])

    ts = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    pretrain_model_name = "%s_%s_%s" % (train_config["pretrain_model"], train_config["predict_net"], ts)
    save_model_dir = train_config["downstream_save_model_dir"]
    save_pretrain_model_dir=train_config["save_pretrain_model_dir"]
    os.makedirs(save_model_dir, exist_ok=True)

    # save config
    with open(os.path.join(save_model_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f)

    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(os.path.join(save_model_dir, "train_log.txt"), 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    # set device
    device = torch.device("cuda:%d" % train_config["gpu_id"] if train_config["gpu_id"] != -1 else "cpu")
    if train_config["gpu_id"] != -1:
        torch.cuda.set_device(device)

    # reset the pattern parameters
    if train_config["share_emb"]:
        train_config["max_npv"], train_config["max_npvl"], train_config["max_npe"], train_config["max_npel"] = \
            train_config["max_ngv"], train_config["max_ngvl"], train_config["max_nge"], train_config["max_ngel"]

    if train_config["pretrain_model"] == "GCN":
        pre_train_model = GCN(train_config)
    if train_config["pretrain_model"] == "GIN":
        pre_train_model = GIN(train_config)
    if train_config["pretrain_model"] == "GAT":
        pre_train_model = GAT(train_config)
    if train_config["pretrain_model"] == "GraphSage":
        pre_train_model = Graphsage(train_config)


    #10fold cross val load data
    dataset = GraphAdjDataset(list())
    dataset.load(os.path.join(train_config["save_data_dir"], "train_dgl_dataset.pt"))
    tenfold = ConsecutiveSplitter.k_fold_split(dataset, k=10)
    acc = list()
    macroF = list()
    weightedF = list()

    for data in tenfold:
        print("--------------------------------------------------------------------------------------")
        print("start new fold")
        np.random.seed()
        torch.random.seed()
        pre_train_model = pre_train_model.to(device)
        pre_train_model.load_state_dict(torch.load(os.path.join(save_pretrain_model_dir, 'best.pt')))
        if train_config["prompt"] == "MEAN":
            model = graph_prompt_layer_mean()
        if train_config["prompt"] == "SUM":
            model = graph_prompt_layer_sum()
        if train_config["prompt"] == "LINEAR-MEAN":
            model = graph_prompt_layer_linear_mean(train_config["gcn_hidden_dim"]*train_config["gcn_graph_num_layers"], train_config["prompt_output_dim"])
        if train_config["prompt"] == "LINEAR-SUM":
            model = graph_prompt_layer_linear_sum(train_config["gcn_hidden_dim"]*train_config["gcn_graph_num_layers"], train_config["prompt_output_dim"])
        if train_config["prompt"] == "FEATURE-WEIGHTED-SUM":
            model = graph_prompt_layer_feature_weighted_sum(train_config["gcn_hidden_dim"]*train_config["gcn_graph_num_layers"])
        if train_config["prompt"] == "FEATURE-WEIGHTED-MEAN":
            model = graph_prompt_layer_feature_weighted_mean(train_config["gcn_hidden_dim"]*train_config["gcn_graph_num_layers"])

        model = model.to(device)
        logger.info(model)
        logger.info("num of parameters: %d" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

        data_loaders = OrderedDict({"train": None, "dev": None})
        data_loaders["train"]=data[0]
        data_loaders["dev"]=data[1]
        for data_type in data_loaders:
            data=GraphAdjDataset_DGL_Input(data_loaders[data_type])

            sampler = Sampler(data, group_by=["graph"], batch_size=train_config["batch_size"],
                              shuffle=data_type == "train", drop_last=False)
            data_loader = DataLoader(data,
                                     batch_sampler=sampler,
                                     collate_fn=GraphAdjDataset.batchify,
                                     pin_memory=data_type == "train")
            data_loaders[data_type] = data_loader
        print('data_loaders', data_loaders.items())

        # optimizer and losses
        writer = SummaryWriter(save_model_dir)
        optimizer = torch.optim.AdamW(itertools.chain(pre_train_model.parameters(),model.parameters()), lr=train_config["lr"],
                                      weight_decay=train_config["weight_decay"], amsgrad=True)

        optimizer.zero_grad()
        scheduler = None
        # scheduler = get_linear_schedule_with_warmup(optimizer,
        #     len(data_loaders["train"]), train_config["epochs"]*len(data_loaders["train"]), min_percent=0.0001)
        best_bp_losses = {"train": INF, "dev": INF, "test": INF}
        best_bp_epochs = {"train": -1, "dev": -1, "test": -1}
        best_acc=0
        best_weightf=0
        best_macrof=0

        total_train_time = 0
        total_dev_time = 0
        total_test_time = 0
        best_c_embedding = None
        c_embedding = None
        for epoch in range(train_config["epochs"]):
            for data_type, data_loader in data_loaders.items():

                if data_type == "train":
                    mean_reg_loss, mean_bp_loss, _time, accfold, c_embedding = train(pre_train_model, model, optimizer, scheduler, data_type, data_loader, device,
                                                               train_config, epoch, logger=logger, writer=writer)
                    total_train_time += _time
                    torch.save(model.state_dict(), os.path.join(save_model_dir, 'epoch%d.pt' % (epoch)))
                    if train_config["update_pretrain"] == True:
                        torch.save(pre_train_model.state_dict(),os.path.join(save_model_dir, 'pretrain_epoch%d.pt' % (epoch)))

                else:
                    mean_reg_loss, mean_bp_loss, evaluate_results, _time,accfold, macroFfold, weightedFfold = evaluate(pre_train_model, model, data_type, data_loader, device,
                                                                                    train_config, epoch, c_embedding, logger=logger,
                                                                                    writer=writer)
                    total_dev_time += _time
                    with open(os.path.join(save_model_dir, '%s%d.json' % (data_type, epoch)), "w") as f:
                        json.dump(evaluate_results, f)

                if mean_bp_loss <= best_bp_losses[data_type]:
                    best_bp_losses[data_type] = mean_bp_loss
                    best_bp_epochs[data_type] = epoch
                    logger.info(
                        "data_type: {:<5s}\tbest mean loss: {:.3f} (epoch: {:0>3d})".format(data_type, mean_bp_loss,
                                                                                       epoch))
                if data_type=="dev" and accfold>=best_acc:
                    best_acc=accfold
                    best_macrof=macroFfold
                    best_weightf=weightedFfold
        acc.append(best_acc)
        macroF.append(best_macrof)
        weightedF.append(best_weightf)

        best_epoch = best_bp_epochs["dev"]
        best_epoch = train_config["epochs"]-1

            #print(model.linear.weight)
            #网络参数没有变化，check参数的梯度变化情况
            #for x in optimizer.param_groups[0]['params']:
            #    print(x.grad)

        for data_type in data_loaders.keys():
            logger.info(
                "data_type: {:<5s}\tbest mean loss: {:.3f} (epoch: {:0>3d})".format(data_type, best_bp_losses[data_type],
                                                                                    best_bp_epochs[data_type]))


    print('acc for 10fold: ', acc)
    print('macroF for 10fold: ', macroF)
    print('weightedF for 10fold: ', weightedF)
    acc=np.array(acc)
    macroF=np.array(macroF)
    weightedF=np.array(weightedF)

    print('acc mean: ',np.mean(acc), 'acc std: ',np.std(acc))
    print('macroF mean: ',np.mean(macroF), 'macroF std: ',np.std(macroF))
    print('weightedF mean: ',np.mean(weightedF), 'weightedF std: ',np.std(weightedF))

    # _________________________________________________________________________________________________________________

