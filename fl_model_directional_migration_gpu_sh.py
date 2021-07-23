__author__ = 'yang.xu'

# Import required libraries
# Primary libraries
import time
import numpy as np
from numpy import linalg
import random
import copy
# from PIL import Image, ImageDraw
# import matplotlib.pyplot as plt
import gc
import resource
import os
import warnings
# warnings.filterwarnings('ignore')
# PyTroch libraries
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import syft as sy # <--NEW: import the PySyft library

from syft.frameworks.torch.fl import utils

# Self-defined libraries
import fl_datasets
import fl_utils
import fl_models
from fl_utils import printer, time_since
from fl_train_test import train, test

import argparse

parser = argparse.ArgumentParser(description='FL-MIGR')

parser.add_argument('--batch_size', type=int, default=50,
                    help='')
parser.add_argument('--vm_test_batch_size', type=int, default=500,
                    help='')
parser.add_argument('--test_batch_size', type=int, default=1000,
                    help='')
parser.add_argument('--enable_vm_test', action="store_true", default=False,
                    help='')
parser.add_argument('--epochs', type=int, default=900,
                    help='')
parser.add_argument('--lr', type=float, default=0.001,
                    help='')
parser.add_argument('--min_lr', type=float, default=0.0001,
                    help='')
parser.add_argument('--vm_num', type=int, default=10,
                    help='')
parser.add_argument('--local_iters', type=int, default=1,
                    help='')
parser.add_argument('--aggregate_interval', type=int, default=20,
                    help='')
parser.add_argument('--dataset_type', type=str, default='FashionMNIST',
                    help='')
parser.add_argument('--model_type', type=str, default='CNN',
                    help='')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='')
parser.add_argument('--no_cuda', action="store_false", default=False,
                    help='')
parser.add_argument('--seed', type=int, default=1,
                    help='')
parser.add_argument('--log_interval', type=int, default=100,
                    help='')
parser.add_argument('--save_model', action="store_true", default=False,
                    help='')
parser.add_argument('--train_flag', action="store_false", default=True,
                    help='')
parser.add_argument('--checkpoint_dir', type=str, default='./Checkpoint_tmp/',
                    help='')
parser.add_argument('--data_pattern_idx', type=int, default=3,
                    help='')
parser.add_argument('--migr_pattern_idx', type=int, default=2,
                    help='')
parser.add_argument('--epoch_start', type=int, default=0,
                    help='')
parser.add_argument('--epoch_step', type=int, default=100,
                    help='')
parser.add_argument('--visible_cuda', type=str, default='2',
                    help='')
parser.add_argument('--enable_lr_decay', action="store_true", default=False,
                    help='')
parser.add_argument('--decay_rate', type=float, default=0.995,
                    help='')
parser.add_argument('--lr_curve_interval', type=int, default=20,
                    help='')
parser.add_argument('--enable_div_out', action="store_true", default=False,
                    help='')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_cuda

use_cuda = not args.no_cuda and torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

torch.manual_seed(args.seed)

device = torch.device('cuda' if use_cuda else 'cpu')

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

if __name__ == "__main__":

    args.lr_curve_interval = args.aggregate_interval

    checkpoint_dir = args.checkpoint_dir
    fl_utils.create_dir(checkpoint_dir)

    # data_pattern_list = ['bias', 'partition', 'random', 'in_cluster','x_cluster','r_cluster']
    data_pattern_list = ['random', 'lowbias', 'midbias', 'highbias']
    data_pattern_idx = args.data_pattern_idx

    migr_pattern_list = ['1-d2d', 'dy-d2d', '1-c2c', 'dy-c2c', 'dy-top2-c2c']
    migr_pattern_idx = args.migr_pattern_idx
    # <--Initialize the param server
    param_server = fl_utils.create_vm_by_id('param_server')

    # <--Initialize the virtual machines
    vm_list = fl_utils.create_vm(args.vm_num)

    print(vm_list)
    
    if args.enable_lr_decay:
        if args.lr <= args.min_lr:
            args.min_lr = args.lr * 0.01
    else:
        args.min_lr = args.lr
    
    LOAD_MODEL_PATH = checkpoint_dir + 'MIGR_GPU' + '_' + args.dataset_type + '_' + args.model_type + '_vm' + str(args.vm_num) + '_' + data_pattern_list[data_pattern_idx] + 'data' + \
        migr_pattern_list[migr_pattern_idx] + '_lr' + str(args.lr) + '_minlr' + str(args.min_lr) + '_decay' + str(args.enable_lr_decay) + '_epoch' + str(args.epochs) + \
        '_local' + str(args.local_iters) + '_vmtest' + str(args.enable_vm_test) + '_agg' + str(args.aggregate_interval) + "_" + str(args.epoch_start) +'.pth'

    SAVE_MODEL_PATH = checkpoint_dir + 'MIGR_GPU' + '_' + args.dataset_type + '_' + args.model_type + '_vm' + str(args.vm_num) + '_' + data_pattern_list[data_pattern_idx] + 'data' + \
        migr_pattern_list[migr_pattern_idx] + '_lr' + str(args.lr) + '_minlr' + str(args.min_lr) + '_decay' + str(args.enable_lr_decay) + '_epoch' + str(args.epochs) + \
        '_local' + str(args.local_iters) + '_vmtest' + str(args.enable_vm_test) + '_agg' + str(args.aggregate_interval) + \
        '_' + str(args.epoch_start + args.epoch_step) +'.pth'
    
    LOG_PATH = checkpoint_dir + 'MIGR_GPU' + '_' + args.dataset_type + '_' + args.model_type + '_vm' + str(args.vm_num) + '_' + data_pattern_list[data_pattern_idx] + 'data' + \
        migr_pattern_list[migr_pattern_idx] + '_lr' + str(args.lr) + '_minlr' + str(args.min_lr) + '_decay' + str(args.enable_lr_decay) + '_epoch' + str(args.epochs) + \
        '_local' + str(args.local_iters) + '_vmtest' + str(args.enable_vm_test) + '_agg' + str(args.aggregate_interval) + '_log.txt' 

    RESULT_PATH = checkpoint_dir + 'result.txt'
    
    fig_dir = checkpoint_dir + 'figure/'
    fl_utils.create_dir(fig_dir)
    FIG_ROOT_PATH = fig_dir + 'MIGR_GPU' + '_' + args.dataset_type + '_' + args.model_type + '_vm' + str(args.vm_num) + '_' + data_pattern_list[data_pattern_idx] + 'data' + \
        '_' + migr_pattern_list[migr_pattern_idx] + '_lr' + str(args.lr) + '_minlr' + str(args.min_lr) + '_decay' + str(args.enable_lr_decay) + '_epoch' + str(args.epochs) + \
        '_local' + str(args.local_iters) + '_vmtest' + str(args.enable_vm_test) + '_agg' + str(args.aggregate_interval) + '_FIGTYPE.png'
    # <--Load datasets
    # if use_cuda:
    #     torch.set_default_tensor_type(torch.FloatTensor)

    train_dataset, test_dataset = fl_datasets.load_datasets(args.dataset_type)

    # <--Create federated train/test loaders for virtrual machines
    if data_pattern_idx == 0:  # random data (IID)
        is_train = True
        vm_train_loaders = fl_utils.create_segment_federated_loader(
            args, kwargs, vm_list, is_train, train_dataset)
        is_train = False
        vm_test_loaders = fl_utils.create_segment_federated_loader(
            args, kwargs, vm_list, is_train, test_dataset)

    else:  # bias data partition (Non-IID)
        if data_pattern_idx == 1:  # lowbias
            label_clusters = ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9))
        elif data_pattern_idx == 2:  # midbias
            label_clusters = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
        elif data_pattern_idx == 3:  # highbias
            label_clusters = ((0,), (1,), (2,), (3,), (4,),
                                (5,), (6,), (7,), (8,), (9,))

        class_num = len(train_dataset.classes)
        cluster_len = len(label_clusters)

        for idx in range(cluster_len):
            train_data_tmp, train_targets_tmp = fl_utils.create_bias_selected_data(
                args, label_clusters[idx], train_dataset)
            test_data_tmp, test_targets_tmp = fl_utils.create_bias_selected_data(
                args, label_clusters[idx], test_dataset)
            if idx == 0:
                train_data = train_data_tmp
                train_targets = train_targets_tmp

                test_data = test_data_tmp
                test_targets = test_targets_tmp
            else:
                train_data = np.vstack((train_data, train_data_tmp))
                train_targets = np.hstack((train_targets, train_targets_tmp))

                test_data = np.vstack((test_data, test_data_tmp))
                test_targets = np.hstack((test_targets, test_targets_tmp))

        new_train_dataset = fl_datasets.train_test_dataset(
            train_data, train_targets, class_num)

        new_test_dataset = fl_datasets.train_test_dataset(
            test_data, test_targets, class_num)

        is_train = True
        vm_train_loaders = fl_utils.create_segment_federated_loader(
            args, kwargs, vm_list, is_train, new_train_dataset)
        is_train = False
        vm_test_loaders = fl_utils.create_segment_federated_loader(
            args, kwargs, vm_list, is_train, new_test_dataset)

    # elif data_pattern_idx >= 3:
    #     label_clusters = ((0,1,2,3,4),(5,6,7,8,9))
    #     data1, targets1 = fl_utils.create_bias_selected_data(args, label_clusters[0], train_dataset)
    #     data2, targets2 = fl_utils.create_bias_selected_data(args, label_clusters[1], train_dataset)
    #     data = np.vstack((data1,data2))
    #     targets = np.hstack((targets1, targets2))
    #     new_train_dataset = fl_datasets.train_test_dataset(data,targets,10)

    #     data1, targets1 = fl_utils.create_bias_selected_data(args, label_clusters[0], test_dataset)
    #     data2, targets2 = fl_utils.create_bias_selected_data(args, label_clusters[1], test_dataset)
    #     data = np.vstack((data1,data2))
    #     targets = np.hstack((targets1, targets2))
    #     new_test_dataset = fl_datasets.train_test_dataset(data,targets,10)

    #     is_train = True
    #     vm_train_loaders = fl_utils.create_segment_federated_loader(
    #         args, kwargs, vm_list, is_train, new_train_dataset)
    #     is_train = False
    #     vm_test_loaders = fl_utils.create_segment_federated_loader(
    #         args, kwargs, vm_list, is_train, new_test_dataset)
    #

    test_loader = fl_utils.create_ps_test_loader(args, kwargs, param_server, test_dataset)

    # if use_cuda:
    #     torch.set_default_tensor_type(torchvm_data2 .cuda.FloatTensor)
    #test_loader = test_loader
    # <--Create Neural Network model instance
    if args.dataset_type == 'FashionMNIST':
        if args.model_type == 'LR':
            model = fl_models.MNIST_LR_Net().to(device)
        else:
            model = fl_models.MNIST_Net().to(device)

    elif args.dataset_type == 'MNIST':
        if args.model_type == 'LR':
            model = fl_models.MNIST_LR_Net().to(device)
        else:
            model = fl_models.MNIST_Small_Net().to(device)
            
    elif args.dataset_type == 'CIFAR10':
        
        if args.model_type == 'Deep':
            model = fl_models.CIFAR10_Deep_Net().to(device)
            args.decay_rate = 0.99
        else:
            model = fl_models.CIFAR10_Net().to(device)
            args.decay_rate = 0.99

    elif args.dataset_type == 'Sent140':
        
        if args.model_type == 'GRU': 
            model = fl_models.Sent140_GRU().to(device)
            args.decay_rate = 1.0
        else:
            model = fl_models.Sent140_RNN().to(device)
            args.decay_rate = 1.0
    elif args.dataset_type == 'CIFAR100':
        model = fl_models.CIFAR100_Net().to(device)
        args.decay_rate = 0.98
    else:
        pass

    # load pretrained model
    epoch_start = args.epoch_start
    epoch_end = epoch_start + args.epoch_step

    log_out = open(LOG_PATH, 'a+')
    result_out = open(RESULT_PATH, 'a+')
    if args.epoch_start == 0:
        log_out.write("%s\n" % LOG_PATH)

    if not args.epoch_start == 0:
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))
    

    aggregate_bandwidth=1
    migration_bandwidth=0.05
    total_bandwidth=0



    # <--Train Neural network and validate with test set after completion of training every epoch
    enable_vm_test = args.enable_vm_test

    weight_cosine_similarity = np.zeros((args.vm_num, args.vm_num))
    vm_migration_idxs = [-1] * args.vm_num
    if migr_pattern_idx == 4:
        K_top = int(np.ceil(0.2 * args.vm_num))
    else:
        K_top = int(np.ceil(0.5 * args.vm_num))

    vm_migration_idxs_array = -1 * np.ones((args.vm_num, K_top))
    MARK = 2
    one_shot_flag = False

    start = time.time()
    for epoch in range(epoch_start + 1, epoch_end + 1):
        args.enable_vm_test = enable_vm_test
        args.train_flag = False
        
        # <--adjust learning rate
        if args.enable_lr_decay:
            vm_lr = np.max((args.decay_rate ** (epoch - 1) * args.lr, args.min_lr)) #<--adjust learning rate
        else:
        	vm_lr = args.lr
        printer("--[Debug] Epoch: {} learning rate: {:.4f}".format(epoch, vm_lr), log_out)

        # imax_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # prnter("--[Debug][EPOCH LOOP] - Memory: {:.2f} MB".format(max_mem_used / 1024), log_out)

        if epoch == epoch_start + 1 or epoch % args.aggregate_interval == 1 or args.aggregate_interval == 1:
            args.train_flag = True
            if not epoch == epoch_start + 1:
                del vm_models
                del vm_optimizers
            vm_models = list()
            vm_optimizers = list()
            for vm_idx in range(args.vm_num):
                # <--distribute the model to all vitrual machines
                vm_model_instance = model.copy().send(vm_list[vm_idx])
                vm_models.append(vm_model_instance)
                # vm_optimizer_instance = optim.RMSprop(params = vm_model_instance.parameters(), lr = vm_lr, alpha = 0.9)
                # <--momentum is not supported at the moment
                vm_optimizer_instance = optim.SGD(
                    params=vm_model_instance.parameters(), lr=vm_lr)
                vm_optimizers.append(vm_optimizer_instance)

            train(args, vm_models, device, vm_train_loaders, vm_test_loaders,
                  vm_optimizers, param_server, epoch, start, log_out)
            
            if one_shot_flag == True:
                pass
            else:
                with torch.no_grad():
                    weight_cosine_similarity = weight_cosine_similarity * 0.0
                    delta_vm_models = list()

                    for vm_idx in range(args.vm_num):
                            vm_models[vm_idx] = vm_models[vm_idx].get()
                            delta_vm_models_tmp = fl_utils.minus_model(
                                    vm_models[vm_idx], model)
                            delta_vm_models.append(np.array(torch.nn.utils.parameters_to_vector(
                                    delta_vm_models_tmp.parameters()).cpu().numpy()))

                    for row_idx in range(args.vm_num):
                        for col_idx in range(args.vm_num):
                            # num = float(delta_vm_models[row_idx].T * delta_vm_models[col_idx])
                            num = delta_vm_models[row_idx].dot(delta_vm_models[col_idx])
                            denom = linalg.norm(delta_vm_models[row_idx]) * linalg.norm(delta_vm_models[col_idx])
                            weight_cosine_similarity[row_idx, col_idx] = num / denom

                    printer('-->[{}] Train Epoch: {} consine similarity: {}'.format(
                                time_since(start), epoch, weight_cosine_similarity), log_out)

                    if migr_pattern_idx == 0 or migr_pattern_idx == 1:
                        if migr_pattern_idx == 0:
                            one_shot_flag = True
                        
                        weight_cosine_similarity_tmp = copy.deepcopy(weight_cosine_similarity)
                        while(weight_cosine_similarity_tmp.min() < MARK):
                            min_sim = weight_cosine_similarity_tmp.min()
                            min_sim_idxs = np.argwhere(weight_cosine_similarity_tmp == min_sim)
                            vm_migration_idxs[min_sim_idxs[0][0]] = min_sim_idxs[0][1]
                            weight_cosine_similarity_tmp[min_sim_idxs[0][0], :] = MARK
                            weight_cosine_similarity_tmp[:, min_sim_idxs[0][1]] = MARK

                    elif migr_pattern_idx == 2 or migr_pattern_idx == 3:
                        if migr_pattern_idx == 2:
                            one_shot_flag = True

                        weight_cosine_similarity_tmp = copy.deepcopy(weight_cosine_similarity)
                        
                        sorted_sim_idxs = np.argsort(weight_cosine_similarity_tmp, axis = 1)
                        vm_migration_idxs_array = sorted_sim_idxs[:, 0:K_top]

                        printer('-->[{}] Train Epoch: {} migration array: {}'.format(
                                time_since(start), epoch, vm_migration_idxs_array), log_out)
                            
                    else:
                        pass

                    for vm_idx in range(args.vm_num):
                        # <--distribute the model to all vitrual machines
                        vm_models[vm_idx] = vm_models[vm_idx].send(vm_list[vm_idx])
                    
        else:
            args.train_flag = True
            vm_models_tmp = list()
            with torch.no_grad():
                for vm_idx in range(args.vm_num):
                    vm_models_tmp.append(vm_models[vm_idx].get())
                    #print('--[Debug] model:', vm_models_tmp[vm_idx])

                if args.enable_div_out:
                    for vm_idx in range(args.vm_num):
                        div_item = fl_utils.calculate_model_divergence(model, vm_models_tmp[vm_idx])
                        div_item_str = dict()
                        for name, _ in div_item.items():
                            div_item_str[name] = div_item[name].item()
                        #div_item_str = [div_item[name].item() for name, _ in div_item.items()]
                        printer('-->[{}] Train Epoch: {} vm: {} div: {}'.format(time_since(start), epoch - 1, vm_idx, str(div_item_str)), log_out)

                print('--[Debug] model aggregation checked!')
                if migr_pattern_idx == 0 or migr_pattern_idx == 1:
                    if np.array(vm_migration_idxs).max() == -1:
                        vm_random_idxs = list(range(args.vm_num))
                        random.shuffle(vm_random_idxs)
                    else:
                        vm_random_idxs = vm_migration_idxs

                elif migr_pattern_idx == 2 or migr_pattern_idx == 3 or migr_pattern_idx == 4:
                    if np.array(vm_migration_idxs_array).max() == -1:
                        vm_random_idxs = list(range(args.vm_num))
                        random.shuffle(vm_random_idxs)
                    else:
                        confirmed_idxs = [-1] * args.vm_num
                        selected_idxs = [-1] * args.vm_num
                        vm_random_idxs = [-1] * args.vm_num

                        for vm_idx in range(args.vm_num):
                            res_idxs1 = np.argwhere(np.array(confirmed_idxs) == -1).reshape(-1)
                            rand_pos = np.random.randint(len(res_idxs1))
                            cur_idxs = vm_migration_idxs_array[rand_pos, :]

                            res_idxs2 = np.argwhere(np.array(selected_idxs) == -1).reshape(-1)

                            idxs_intersection = np.array([idx for idx in cur_idxs if idx in res_idxs2])

                            idxs_complement = np.array([idx for idx in res_idxs2 if idx not in cur_idxs])

                            if len(idxs_intersection) > 0:
                                pos = np.random.randint(len(idxs_intersection))

                                vm_random_idxs[res_idxs1[rand_pos]] = idxs_intersection[pos]
                                selected_idxs[idxs_intersection[pos]] = 1
                            else:
                                pos = np.random.randint(len(idxs_complement))

                                vm_random_idxs[res_idxs1[rand_pos]] = idxs_complement[pos]
                                selected_idxs[idxs_complement[pos]] = 1

                            confirmed_idxs[res_idxs1[rand_pos]] = 1
                            

                printer('--[{}] directional idxs: {}'.format(time_since(start), vm_random_idxs), log_out)
                
                if data_pattern_idx == 1:
                    for vm_idx in range(args.vm_num):
                        if vm_idx<=4 and vm_random_idxs[vm_idx]>4:
                            total_bandwidth=total_bandwidth+migration_bandwidth
                        if vm_idx>4 and vm_random_idxs[vm_idx]<=4:
                            total_bandwidth=total_bandwidth+migration_bandwidth
                elif data_pattern_idx == 2:
                    for vm_idx in range(args.vm_num):
                        if vm_idx%2==0:
                            if vm_idx!=vm_random_idxs[vm_idx] and vm_idx!=vm_random_idxs[vm_idx+1]:
                                total_bandwidth=total_bandwidth+migration_bandwidth
                        else:
                            if vm_idx!=vm_random_idxs[vm_idx] and vm_idx!=vm_random_idxs[vm_idx-1]:
                                total_bandwidth=total_bandwidth+migration_bandwidth
                elif data_pattern_idx == 3:
                    for vm_idx in range(args.vm_num):
                        if vm_idx!=vm_random_idxs[vm_idx]:
                            total_bandwidth=total_bandwidth+migration_bandwidth




            for vm_idx in range(args.vm_num):
                vm_model_instance = vm_models_tmp[vm_idx].copy().send(vm_list[vm_random_idxs[vm_idx]])# <--very important
                vm_models[vm_random_idxs[vm_idx]] = vm_model_instance
                # vm_optimizer_instance = optim.RMSprop(params = vm_model_instance.parameters(), lr = vm_lr, alpha = 0.9)
                # <--momentum is not supported at the moment
                vm_optimizer_instance = optim.SGD(params=vm_model_instance.parameters(), lr=vm_lr)
                vm_optimizers[vm_random_idxs[vm_idx]] = vm_optimizer_instance  # <--very important

            del vm_models_tmp

            print('--[Debug] model exchange checked!')
            train(args, vm_models, device, vm_train_loaders, vm_test_loaders,
                  vm_optimizers, param_server, epoch, start, log_out)

        # if epoch == args.epochs:
        if epoch % args.aggregate_interval == 0: 
            with torch.no_grad():
                # <--seperately sum up params of each layer of CNN of all vitrual machines
                update_model = []
                for vm_idx in range(args.vm_num):
                    vm_models[vm_idx] = vm_models[vm_idx].get()
                    if args.enable_div_out:
                        div_item = fl_utils.calculate_model_divergence(model, vm_models[vm_idx])
                        div_item_str = dict()
                        for name, _ in div_item.items():
                            div_item_str[name] = div_item[name].item()
                        #div_item_str = [div_item[name].item() for name, _ in div_item.items()]
                        printer('-->[{}] Train Epoch: {} vm: {} div: {}'.format(time_since(start), epoch, vm_idx, str(div_item_str)), log_out)

                for vm_idx in range(args.vm_num):
                    total_bandwidth=total_bandwidth+aggregate_bandwidth
                    if vm_idx == 0:
                        update_model = vm_models[vm_idx]
                    else:
                        update_model = fl_utils.add_model(update_model, vm_models[vm_idx])

                model = fl_utils.scale_model(update_model, 1.0 / args.vm_num)
                del update_model

            args.enable_vm_test = False
            ps_model = model.copy().send(param_server)
            test(args, ps_model, device, test_loader, epoch, start, log_out, total_bandwidth, result_out)
            del ps_model

            if (args.save_model):
                # pass
                torch.save(model.state_dict(), SAVE_MODEL_PATH)
        
        gc.collect()

        if args.lr_curve_interval > 0:
            if epoch % args.lr_curve_interval == 0:
                fl_utils.plot_learning_curve(LOG_PATH, FIG_ROOT_PATH)

    log_out.close()
    result_out.close()
