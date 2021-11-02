#!/usr/bin/env python
import torch
import sys
import time
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np
import torchvision
import torch.nn.functional as F
import random
import os
import socket
import time
import struct
import argparse
from util.utils import send_msg, recv_msg, time_printer, time_count
import copy
from torch.autograd import Variable
from model.model_VGG_cifar import construct_VGG_cifar
from model.model_AlexNet_cifar import construct_AlexNet_cifar
from model.model_nin_cifar import construct_nin_cifar
from model.model_VGG_image import construct_VGG_image
from model.model_AlexNet_image import construct_AlexNet_image
from model.model_nin_image import construct_nin_image
from util.utils import printer
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='PyTorch MNIST SVM')
parser.add_argument('--device_num', type=int, default=3, metavar='N',
                        help='number of working devices (default: 3)')
parser.add_argument('--node_num', type=int, default=1, metavar='N',
                        help='device index (default: 1)')
parser.add_argument('--device_ip', type=str, default='localhost', metavar='N',
                        help=' ip address')
parser.add_argument('--device_port', type=int, default='50001', metavar='N',
                        help=' ip port')
parser.add_argument('--use_gpu', type=int, default=0, metavar='N',
                        help=' ip port')
parser.add_argument('--device_ip_list', type=list, default=['localhost'], metavar='N',
                        help=' ip port')
parser.add_argument('--model_type', type=str, default='NIN', metavar='N',          #NIN,AlexNet,VGG
                        help='model type')
parser.add_argument('--dataset_type', type=str, default='cifar10', metavar='N',  #cifar10,cifar100,image
                        help='dataset type')
args = parser.parse_args()

if args.use_gpu == 0:
    print('use gpu')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)
#torch.cuda.manual_seed(args.seed) #<--random seed for one GPU
#torch.cuda.manual_seed_all(args.seed) #<--random seed for multiple GPUs
device_gpu = torch.device("cuda" if args.use_gpu == 0 else "cpu")
# Configurations are in a separate config.py file

sock_ps = socket.socket()
#sock_ps.connect(('localhost', 50010))
sock_ps.connect(('172.16.50.10', 50010))

sock_edge1 = socket.socket()
sock_edge1.connect(('192.168.0.127', 51001))
sock_edge = []
sock_edge.append(sock_edge1)

sock_edge2 = socket.socket()
sock_edge2.connect(('192.168.0.127', 51002))
sock_edge = []
sock_edge.append(sock_edge2)

sock_edge3 = socket.socket()
sock_edge3.connect(('192.168.0.102', 51003))
sock_edge = []
sock_edge.append(sock_edge3)

print('---------------------------------------------------------------------------')


if args.dataset_type == 'cifar100':
    transform = transforms.Compose([ 
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                ])
    trainset = datasets.CIFAR100('/data/zywang/Dataset', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)

elif args.dataset_type == 'cifar10':
    transform = transforms.Compose([ 
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                ])
   # trainset = datasets.CIFAR10('/data/zywang/Dataset/cifar10', download=True, train=True, transform=transform)
    trainset = datasets.ImageFolder('/data/zywang/Dataset/cifar_coopfl/train', transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

elif args.dataset_type == 'image' and args.model_type != "AlexNet":
    transform = transforms.Compose([  transforms.Resize((224,224)),
                               #   transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                              ])
    trainset = datasets.ImageFolder('/data/zywang/Dataset/IMAGE10/test', transform = transform)
    #trainset = datasets.ImageFolder('/data/zywang/PartImagenet/train', transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

elif args.dataset_type == 'image' and args.model_type == "AlexNet":
    transform = transforms.Compose([  transforms.Resize((227,227)),
                               #   transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                              ])
    trainset = datasets.ImageFolder('/data/zywang/Dataset/IMAGE10/test', transform = transform)
    #trainset = datasets.ImageFolder('/data/zywang/PartImagenet/train', transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)


device_num = args.device_num
node_num = args.node_num
lr=0.01


criterion = nn.NLLLoss()
local_update = 10
local_iter = 10
first_epoch_flag = True


def local_train(sock, models, partition_way):
    global first_epoch_flag
    global compute_device
    if first_epoch_flag ==True:
        first_epoch_flag = False
        local_update = 1
    else:
        local_update = 5
    batch = [None]*local_iter
    a= []
    for i in range(len(trainloader)):
        a.append(i)
 #   print(a)
    random.shuffle(a)
  #  print(a)
    for i in range(len(batch)):
        batch[i] = a[i]
    print(batch)
    for ep in range(1):
        for i in range(0, len(partition_way)):
            if partition_way[i]==0:
                models[i].train()
        count=0
    # forward
        for images, labels in trainloader:
            count+=1
            if count in batch:
                print("batch_training"+str(count))
            else:
                continue
            images= images.to(device_gpu)
            if partition_way[len(partition_way)-1] == 0:
                labels = labels.to(device_gpu)
            else: 
                fob=2 #label
                msg = ['CLIENT_TO_SERVER',ep,fob,node_num,0,labels]
                send_msg(sock, msg)
            start_time = time.time()
            input[0] = images
            output[0] = models[0](input[0])
            input[1] = output[0].detach().requires_grad_()
            end_time = time.time()
            if local_update == 1:
                compute_device[0][0] += time_count(start_time, end_time)
      #      time_printer(start_time,end_time,input[0],0,1)
            for i in range(1,len(models)):

                if partition_way[i]==0 and partition_way[i-1]==0:
                    start_time = time.time()
                    output[i] = models[i](input[i])
                    end_time= time.time()
                    if local_update == 1:
                        compute_device[0][i] += time_count(start_time, end_time)
                 #   time_printer(start_time,end_time,input[i],i,1)
                    if i<len(models)-1:
                        input[i+1] = output[i].detach().requires_grad_()
                    else:
                        loss = criterion(output[i], labels)

                elif partition_way[i]==0 and partition_way[i-1]==1:
                    msg = recv_msg(sock,'SERVER_TO_CLIENT')
                    start_time = time.time()
                 #   print(msg[1])
                    input[i] = msg[1].to(device_gpu)
            #        input[current_layer] = input[current_layer].detach().requires_grad_()
                    input[i] = input[i].detach().requires_grad_(True)
                  #  print(input[i])
                    output[i] = models[i](input[i])
                    end_time = time.time()
                    if local_update == 1:
                        compute_device[0][i] += time_count(start_time, end_time)
            #        time_printer(start_time,end_time,input[i],i,1)
                    if i<len(models)-1:
                        input[i+1] = output[i].detach().requires_grad_()
                    else:
                        loss = criterion(output[i], labels)

                elif partition_way[i]==1 and partition_way[i-1]==0:
                    fob = 0 #forward
                    msg = ['CLIENT_TO_SERVER',ep,fob,node_num,i,input[i]]
                    send_msg(sock, msg)
                elif partition_way[i]==1 and partition_way[i-1]==1:
                    pass

    #梯度初始化为zero
            for i in range(0,len(partition_way)):
                if optimizers[i] !=None and partition_way[i]==0:
                    optimizers[i].zero_grad()
    #回传
            if partition_way[len(models)-1]==0:
                start_time = time.time()
                loss.backward()
                end_time = time.time()        
                compute_device[1][len(compute_device)-1] += time_count(start_time, end_time)
           #     time_printer(start_time,end_time,input[len(models)-1],len(models),0)

            for i in range(len(models)-2, -1, -1):
                if partition_way[i]==0 and partition_way[i+1]==0:
                    start_time = time.time()
                    grad_in = input[i+1].grad
                    output[i].backward(grad_in)
                    end_time = time.time()
                    if local_update == 1:
                        compute_device[1][i] += time_count(start_time, end_time)
                #    compute_device[i] += time_count(start_time, end_time)
          #          time_printer(start_time,end_time,grad_in,i,0)
                elif partition_way[i]==0 and partition_way[i+1]==1:
                    msg = recv_msg(sock,'SERVER_TO_CLIENT')
                    start_time=time.time()
                    grad_in = msg[1].to(device_gpu)
                    output[i].backward(grad_in)
                    end_time = time.time()
                    if local_update == 1:
                        compute_device[1][i] += time_count(start_time, end_time)
                  #  compute_device[i] += time_count(start_time, end_time)
            #        time_printer(start_time,end_time,grad_in,i,0)
                elif partition_way[i]==1 and partition_way[i+1]==0:
                    grad_in = input[i+1].grad
                    fob = 1 #backward
                    msg = ['CLIENT_TO_SERVER',ep,fob,node_num,i,grad_in]
                    send_msg(sock, msg)
    #更新参数              
            for i in range(0,len(partition_way)):
                if optimizers[i] !=None:
                    optimizers[i].step()
            
        if local_update ==1:
            for  i in range(len(compute_device)):
                for  j in range(len(compute_device[i])):
                    compute_device[i][j] = compute_device[i][j]/(local_iter)

    msg = ['CLIENT_TO_SERVER',ep,4,node_num,0,models]
    send_msg(sock, msg)
    # test(models, testloader, "Test set", ep)
'''
compute_device = [[82,30,142,17,53,100,52,4,19,10,0],
                    [95,16,343,7,105,147,111,1,62,31,3]]
                    '''
model_length = 0
if args.dataset_type == "image":
    if args.model_type == "NIN":
        model_length = 16
    elif args.model_type == "AlexNet":
        model_length = 11
    elif args.model_type == "VGG":
        model_length = 21
else:
    if args.model_type == "NIN":
        model_length = 12
    elif args.model_type == "AlexNet":
        model_length = 11
    elif args.model_type == "VGG":
        model_length = 21


compute_device = []
for i in range(2):
    compute_device.append([])
    for j in range(model_length):
        compute_device[i].append(0)
communication_device = [100/8,100/8]#the bandwidth between devive and edges
memory_device = 8*1024
data_size = len(trainloader.dataset)

while True:
    print(compute_device,communication_device)
    msg = ['CLIENT_TO_SERVER',compute_device,communication_device,memory_device,data_size]
    send_msg(sock_ps,msg)
    msg = recv_msg(sock_ps,'SERVER_TO_CLIENT')
   # global_model = msg[1]
    partition_way = msg[1]
    offloading_descision = msg[2]
 #   if offloading_descision[node_num-1] == 1:
    msg = recv_msg(sock_edge[offloading_descision[node_num-1]-1], 'SERVER_TO_CLIENT')
    model_size = 0
    for model in msg[1]:
        if model!=None:
            for para in model.parameters():
                model_size+=sys.getsizeof(para.storage())/(1024*1024)
    if model_size>5:
        communication_device[0] = model_size/ (time.time()-msg[2])
    print(communication_device,model_size,time.time()-msg[2])
    global_model = msg[1]
    print(partition_way)
    if args.dataset_type == "image":
        if args.model_type == "NIN":
            models, optimizers = construct_nin_image([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
        elif args.model_type == "AlexNet":
            models, optimizers = construct_AlexNet_image([0,0,0,0,0,0,0,0,0,0,0],lr)
        elif args.model_type == "VGG":
            models, optimizers = construct_VGG_image([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
    else:
        if args.model_type == "NIN":
            models, optimizers = construct_nin_cifar([0,0,0,0,0,0,0,0,0,0,0,0],lr)
        elif args.model_type == "AlexNet":
            models, optimizers = construct_AlexNet_cifar([0,0,0,0,0,0,0,0,0,0,0],lr)
        elif args.model_type == "VGG":
            models, optimizers = construct_VGG_cifar([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)

 #   models, optimizers = construct_resnet(partition_way,lr)
    for i in range(len(global_model)):
        if partition_way[i] == 0:
            models[i] = copy.deepcopy(global_model[i])
            models[i] = models[i].to(device_gpu)
    for i in range(len(optimizers)):
        if optimizers[i]!=None and partition_way[i]==0:
            optimizers[i] = optim.SGD(params = models[i].parameters(), lr = lr)
    input=[None]*len(partition_way)
    output=[None]*len(partition_way)
 #   if offloading_descision[node_num-1] == 1:
    local_train(sock_edge[offloading_descision[node_num-1]-1], models, partition_way)
    # elif offloading_descision[node_num-1] ==2:
    #     local_train(sock_edge2, models, partition_way)

