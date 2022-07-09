################################################
'''
This file is to construct the geo_CGNN model and finish the training process
necessary input:

--cutoff, default=8A: It defines the radius of the neighborhood  
--max_nei, default=12 : The number of max neighbors of each node
--lr, default=8e-3 : Learning rate
--test_ratio, default=0.2 : The ratio of test set and validate set
--num_epochs, default=5 : The number of epochs 
--dataset_path, default='database' : The root of dataset
--datafile_name, default="my_graph_data_OQMD_8_12_100" : The first X letters of the data file name
--database, default="OQMD" : The file name of the target
--target_name, default='formation_energy_per_atom' : target name

output:
trained model will output to "./model"
training history/ test predictions / graph vector of test data will output to "./data"
'''
################################################
from dataclasses import dataclass
from posixpath import split
import time
import json
import os
import copy
import numpy as np
import pandas as pd
import random
from sklearn.metrics import r2_score, mean_absolute_error
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from models import Model, geo_CGNN
from data_utils import AtomGraphDataset, Atomgraph_collate


def TransferLearning(model, pre_model_name, n_conv, n_MLP_psi2n=None, n_gated_pooling=None, n_linear_regression=None, conv_TL=True, MLP_psi2n_TL=False, gated_pooling_TL=False, linear_regression_TL=False):
    path = os.getcwd()
    PreModel_path = path + '//model//' + pre_model_name
    state_dict = model.model.state_dict().copy()
    PreModel = torch.load(PreModel_path, map_location=lambda storage, loc: storage)
    state_dict_TL = PreModel.copy()

    # Embedding layer
    state_dict['embedding.linear.weight'] = state_dict_TL['embedding.linear.weight']
    if conv_TL:
        for conv in range(n_conv):
            # in the nodes, for every conv, including linear_gate(), activation_gate(), linear_MLP(), activation_MLP()
            state_dict['conv.{}.linear_gate.weight'.format(conv)] = state_dict_TL['conv.{}.linear_gate.weight'.format(conv)]
            state_dict['conv.{}.activation_gate.0.weight'.format(conv)] = state_dict_TL['conv.{}.activation_gate.0.weight'.format(conv)]
            state_dict['conv.{}.activation_gate.0.bias'.format(conv)] = state_dict_TL['conv.{}.activation_gate.0.bias'.format(conv)]
            state_dict['conv.{}.linear_MLP.weight'.format(conv)] = state_dict_TL['conv.{}.linear_MLP.weight'.format(conv)]
            state_dict['conv.{}.activation_MLP.0.weight'.format(conv)] = state_dict_TL['conv.{}.activation_MLP.0.weight'.format(conv)]
            state_dict['conv.{}.activation_MLP.0.bias'.format(conv)] = state_dict_TL['conv.{}.activation_MLP.0.bias'.format(conv)]
            # in the combine_set, for every conv, including linear1_vector()
            state_dict['conv.{}.linear1_vector.weight'.format(conv)] = state_dict_TL['conv.{}.linear1_vector.weight'.format(conv)]
            #in the plane_wave, for every conv, including linear2_vector_gate(), activation2_vector_gate(), linear2_vector()
            state_dict['conv.{}.linear2_vector_gate.weight'.format(conv)] = state_dict_TL['conv.{}.linear2_vector_gate.weight'.format(conv)]
            state_dict['conv.{}.activation2_vector_gate.0.weight'.format(conv)] = state_dict_TL['conv.{}.activation2_vector_gate.0.weight'.format(conv)]
            state_dict['conv.{}.activation2_vector_gate.0.bias'.format(conv)] = state_dict_TL['conv.{}.activation2_vector_gate.0.bias'.format(conv)]
            state_dict['conv.{}.linear2_vector.weight'.format(conv)] = state_dict_TL['conv.{}.linear2_vector.weight'.format(conv)]
        print("-----------")
        print("conv transfer learning has been finished!")
        print("-----------")
    '''
    if MLP_psi2n_TL:
        for MLP_psi2n in range(n_MLP_psi2n):
        
    if  gated_pooling_TL:
        for gated_pooling in range(n_gated_pooling):
            
    if linear_regression_TL:
        for linear_regression in range(n_linear_regression):
    '''
    model.model.load_state_dict(state_dict)
    print("model has been matched !")




def use_setpLR(param):
    ms = param["milestones"]
    return ms[0] < 0


def create_model(device, model_param, optimizer_param, scheduler_param, load_model, model_name):
    model = geo_CGNN(**model_param)
    if load_model:  # transfer learning
        for para in model.embedding.parameters():
            para.requires_grad = False
        for para in model.conv.parameters():
            para.requires_grad = False
        for para in model.MLP_psi2n.parameters():
            para.requires_grad = False
        print("Freezed embedding/conv/MLP_psi2n")

    clip_value = optimizer_param.pop("clip_value")
    optim_name = optimizer_param.pop("optim")
    if optim_name == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), momentum=0.9,
                              nesterov=True, **optimizer_param)
    elif optim_name == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_param)
    elif optim_name == "amsgrad":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), amsgrad=True,
                               **optimizer_param)
    elif optim_name == "adagrad":
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr_decay=0.1,
                                  **optimizer_param)
    else:
        raise NameError("optimizer {} is not supported".format(optim_name))
    use_cosine_annealing = scheduler_param.pop("cosine_annealing")
    if use_cosine_annealing:
        params = dict(T_max=scheduler_param["milestones"][0],
                      eta_min=scheduler_param["gamma"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)

    elif use_setpLR(scheduler_param):
        scheduler_param["step_size"] = abs(scheduler_param["milestones"][0])
        scheduler_param.pop("milestones")
        scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_param)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_param)
    N_block = model_param.pop('N_block')
    cutoff = model_param.pop('cutoff')
    max_nei = model_param.pop('max_nei')
    name = str(N_block) + '_' + str(cutoff) + '_' + str(max_nei)
    return Model(device, model, model_name, name, optimizer, scheduler, clip_value)


def K_fold(dataset, seed, cv=5, dataloader_param=None, pin_memory=False):
    n_graph = len(dataset.graph_data)
    random.seed(seed)
    indices = list(range(n_graph))
    random.shuffle(indices)
    total_size = len(dataset)
    test_ratio = float(1 / cv)
    test_size = int(test_ratio * total_size)
    if total_size % cv == 0:
        step = int(total_size / cv)
    else:
        step = int(total_size / cv) + 1
    train_data = []
    test_data = []
    split_data = []
    for i in range(cv):
        split = {"test": indices[i * step:i * step + step],
                 "train": indices[0:i * step] + indices[i * step + step:total_size]}

        split_data.append(split)

        test_sampler = SubsetRandomSampler(split['test'])

        train_sampler = SubsetRandomSampler(split['train'])

        test_loader = DataLoader(dataset, sampler=test_sampler, **dataloader_param, pin_memory=pin_memory)
        train_loader = DataLoader(dataset, sampler=train_sampler, **dataloader_param, pin_memory=pin_memory)
        train_data.append(train_loader)
        test_data.append(test_loader)
    return train_data, test_data, split_data


def main(device, model_param, optimizer_param, scheduler_param, dataset_param, dataloader_param,
         num_epochs, seed, load_model, pred, pre_trained_model_path, TL, pre_model_name, model_name):
    N_block = model_param['N_block']
    cutoff = model_param['cutoff']
    max_nei = model_param['max_nei']
    print("Seed:", seed)
    print()
    torch.manual_seed(seed)
    # Create dataset
    dataset = AtomGraphDataset(dataset_param["dataset_path"], dataset_param['datafile_name'], dataset_param["database"],
                               dataset_param["target_name"], model_param['cutoff'], model_param['N_shbf'],
                               model_param['N_srbf'], model_param['n_grid_K'], model_param['n_Gaussian'])

    dataloader_param["collate_fn"] = Atomgraph_collate

    random.seed(seed)
    # 5 fold cross validation
    train_data, test_data, split_data = K_fold(dataset=dataset, seed=seed, dataloader_param=dataloader_param)
    print('5 fold cross validation has been succeed!')
    # every fold to reset the file
    name = str(N_block) + '_' + str(cutoff) + '_' + str(max_nei)
    with open("data/test_predictions_{}.csv".format(name), 'w') as f1:
        f1.truncate()
        f1.close()
    with open("data/all_graph_vec_{}.csv".format(name), 'w') as f2:
        f2.truncate()
        f2.close()

    for cv in range(5):
        # Create a DFTGN model
        # recreate a new model
        model_param['n_node_feat'] = dataset.graph_data[0].nodes.shape[1]
        model = create_model(device, model_param, optimizer_param, scheduler_param, load_model, model_name)

        if TL:
            TransferLearning(model, pre_model_name, n_conv=options['N_block'], conv_TL=True)

        if load_model:
            print("Loading weights from mymodel.pth")
            model.load(model_path=pre_trained_model_path)
            print("Model loaded at: {}".format(pre_trained_model_path))

        if not pred:
            # Train
            train_dl = train_data[cv]
            val_dl = test_data[cv]
            test_dl = test_data[cv]
            split_dl = split_data[cv]
            print(split_dl)
            trainD = [n for n in train_dl]

            print('start training')
            model.train(train_dl, val_dl, num_epochs)

            if num_epochs > 0:
                model.save()

        # Test

        outputs, targets, all_graph_vec = model.evaluate(test_dl)
        names = [dataset.graph_names[i] for i in split_data[cv]["test"]]
        df_predictions = pd.DataFrame({"name": names, "prediction": outputs, "target": targets})
        all_graph_vec = pd.DataFrame(all_graph_vec)
        all_graph_vec['name'] = names
        df_predictions.to_csv("data/test_predictions_{}.csv".format(name), index=False, mode='a', header=None)
        all_graph_vec.to_csv("data/all_graph_vec_{}.csv".format(name), index=False, mode='a')
        print("\nfold {} END".format(cv))
        model_param = {k: options[k] for k in model_param_names if options[k] is not None}
        optimizer_param = {k: options[k] for k in optimizer_param_names if options[k] is not None}
        if optimizer_param["clip_value"] == 0.0:
            optimizer_param["clip_value"] = None
        scheduler_param = {k: options[k] for k in scheduler_param_names if options[k] is not None}
        del model
    file = pd.read_csv("data/test_predictions_{}.csv".format(name), header=None)
    predictons = file[1]
    sorce_targets = file[2]
    mae = mean_absolute_error(sorce_targets, predictons)
    r2 = r2_score(sorce_targets, predictons)
    print("^^^^mae: ", mae)
    print("^^^^r2: ", r2)
    print("\nEND")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Crystal Graph Neural Networks")

    parser.add_argument("--n_hidden_feat", type=int, default=128,
                        help='the dimension of node features')
    parser.add_argument("--conv_bias", type=bool, default=False,
                        help='use bias item or not in the linear layer')
    parser.add_argument("--n_GCN_feat", type=int, default=128)
    parser.add_argument("--N_block", type=int, default=6)
    parser.add_argument("--N_shbf", type=int, default=6)
    parser.add_argument("--N_srbf", type=int, default=6)
    parser.add_argument("--cutoff", type=int, default=8)
    parser.add_argument("--max_nei", type=int, default=12)
    parser.add_argument("--n_MLP_LR", type=int, default=3)
    parser.add_argument("--n_grid_K", type=int, default=4)
    parser.add_argument("--n_Gaussian", type=int, default=64)
    parser.add_argument("--node_activation", type=str, default="Sigmoid")
    parser.add_argument("--MLP_activation", type=str, default="Elu")
    parser.add_argument("--use_node_batch_norm", type=bool, default=True)
    parser.add_argument("--use_edge_batch_norm", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=8e-3)
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--clip_value", type=float, default=0)
    parser.add_argument("--milestones", nargs='+', type=int, default=[20])
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--cosine_annealing", action='store_true')
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--dataset_path", type=str, default='database')
    parser.add_argument("--datafile_name", type=str, default="my_graph_data_OQMD_8_12_100")
    parser.add_argument("--database", type=str, default="OQMD")
    parser.add_argument("--target_name", type=str, default='formation_energy_per_atom')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--load_model", action='store_true')
    parser.add_argument("--pred", action='store_true')
    parser.add_argument("--TL", action='store_true')
    parser.add_argument("--pre_model_name", type=str, default='model_5_8_12.pth')
    parser.add_argument("--model_name", type=str, default='thermal.pth')
    parser.add_argument("--pre_trained_model_path", type=str, default='./pre_trained/model_Ef_OQMD.pth')
    options = vars(parser.parse_args())

    # set cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Model parameters
    model_param_names = ['n_hidden_feat', 'conv_bias', 'n_GCN_feat', 'N_block', 'N_shbf', 'N_srbf', 'cutoff', 'max_nei',
                         'n_MLP_LR', 'node_activation', 'MLP_activation', 'use_node_batch_norm', 'use_edge_batch_norm',
                         'n_grid_K', 'n_Gaussian']
    model_param = {k: options[k] for k in model_param_names if options[k] is not None}
    if model_param["node_activation"].lower() == 'none':
        model_param["node_activation"] = None
    if model_param["MLP_activation"].lower() == 'none':
        model_param["MLP_activation"] = None
    print("Model_param:", model_param)
    print()

    # Optimizer parameters
    optimizer_param_names = ["optim", "lr", "weight_decay", "clip_value"]
    optimizer_param = {k: options[k] for k in optimizer_param_names if options[k] is not None}
    if optimizer_param["clip_value"] == 0.0:
        optimizer_param["clip_value"] = None
    print("Optimizer:", optimizer_param)
    print()

    # Scheduler parameters
    scheduler_param_names = ["milestones", "gamma", "cosine_annealing"]
    # scheduler_param_names = ["milestones", "gamma"]
    scheduler_param = {k: options[k] for k in scheduler_param_names if options[k] is not None}
    print("Scheduler:", scheduler_param)
    print()

    # Dataset parameters
    dataset_param_names = ["dataset_path", 'datafile_name', 'database', "target_name", "test_ratio"]
    dataset_param = {k: options[k] for k in dataset_param_names if options[k] is not None}
    print("Dataset:", dataset_param)
    print()

    # Dataloader parameters
    dataloader_param_names = ["num_workers", "batch_size"]
    dataloader_param = {k: options[k] for k in dataloader_param_names if options[k] is not None}
    print("Dataloader:", dataloader_param)
    print()

    main(device, model_param, optimizer_param, scheduler_param, dataset_param, dataloader_param,
         options["num_epochs"], options["seed"], options["load_model"], options["pred"],
         options["pre_trained_model_path"], options["TL"], options["pre_model_name"], options['model_name'])

