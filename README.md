We provide the code (in pytorch) and datasets for our paper [**"Generalized Graph Prompt: Toward a Unification of Pre-Training and Downstream Tasks on Graphs"**](https://arxiv.org/pdf/2302.08043.pdf),
which is an extension of [**"GraphPrompt: Unifying Pre-Training and Downstream Tasks for Graph Neural Networks"**](https://dl.acm.org/doi/pdf/10.1145/3543507.3583386) accepted by the ACM Web Conference (WWW) 2023.

## Description
- **data/**: contains data we use.
- **graphdownstream/**: implements pre-training and downstream tasks at the graph level.
- **nodedownstream/**: implements downstream tasks at the node level.


## Package Dependencies 

1.  3.6.0<= python <=3.8.0
2.  pip install -r requirements.txt

## Getting Started 
### Graph Classification

Default dataset is ENZYMES. You need to change the corresponding parameters in *pre_train.py* and *prompt_fewshot.py* to train and evaluate on other datasets.

Pretrain:
```sh
-python pre_train_GP.py --model GIN --gpu_id 0 --gcn_hidden_dim 32 --temperature 0.2 --batch_size 1024 --pretrain_hop_num 0 --lr 0.1 --epochs 400 --dropout 0 --seed 0 --max_ngv 126 --max_nge 298 --max_ngvl 7 --max_ngel 2 --node_feature_dim 18 --graph_label_num 6 --graph_dir ../data/ENZYMES/raw --graphslabel_dir ../data/ENZYMES/ENZYMES_graph_labels.txt --save_data_dir ../data/ENZYMESPreTrain --save_model_dir ../dumps/ENZYMESPreTrain/GIN --share_emb False --predict_net_add_enc True --predict_net_add_degree True
```
Prompt tune and test:

```sh
python prompt_fewshot_GP.py --pretrain_model GIN --gpu_id 0 --reg_loss NLL --bp_loss NLL --prompt FEATURE-WEIGHTED-SUM --epochs 100 --lr 0.01 --update_pretrain False --seed 0 --dropout 0 --dataset_seed 0 --train_shotnum 5 --val_shotnum 5 few_shot_tasknum 100 --gcn_graph_num_layers 3 --gcn_hidden_dim 32 --graph_finetuning_output_dim 2 --batch_size 512 --max_ngv 126 --max_nge 298 --max_ngvl 7 --max_ngel 2 --node_feature_dim 18 --graph_label_num 6 --graph_dir ../data/ENZYMES/raw --graphslabel_dir ../data/ENZYMES/ENZYMES_graph_labels.txt --save_data_dir ../data/ENZYMESPreTrain --save_pretrain_model_dir ../dumps/ENZYMESPreTrain/GIN --downstream_save_model_dir ../dumps/ENZYMESGraphClassification/Prompt/GIN-FEATURE-WEIGHTED-SUM/5train5val100task --save_fewshot_dir ../data/ENZYMESGraphClassification/fewshot --share_emb False --predict_net_add_enc True --predict_net_add_degree True
```


### Node Classification

Default dataset is ENZYMES. You need to change the corresponding parameters in *prompt_fewshot.py* to train and evaluate on other datasets. 
```sh
python run_mix_GP.py --pretrain_model GIN --gpu_id 0 --reg_loss NLL --bp_loss NLL --prompt FEATURE-WEIGHTED-SUM --epochs 100 --lr 0.1 --update_pretrain False --seed 0 --dropout 0 --dataset_seed 0 --train_shotnum 1 --val_shotnum 1 few_shot_tasknum 10 --nhop_neighbour 1 --gcn_graph_num_layers 3 --gcn_hidden_dim 32 --prompt_output_dim 2 --batch_size 1024 --max_ngv 126 --max_nge 282 --max_ngvl 3 --max_ngel 2 --node_feature_dim 18 --graph_label_num 6 --graph_num 53 --graph_dir ../data/ENZYMES/allraw --save_data_dir ../data/ENZYMES/all --save_pretrain_model_dir ../dumps/ENZYMESPreTrain/GIN --downstream_save_model_dir ../dumps/ENZYMESNodeClassification/Prompt/GIN-FEATURE-WEIGHTED-SUM/all/1train1val10task --save_fewshot_dir ../data/ENZYMES/nodefewshot --process_raw False --split False
```
