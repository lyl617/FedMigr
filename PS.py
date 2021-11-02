#!/usr/bin/env python
import socket
import time
import torch
import sys
import time
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np
import argparse
import torchvision
import torch.nn.functional as F
import random
import os
import socket
import threading
import time
import struct
from util.utils import send_msg, recv_msg, time_printer,add_model, scale_model, printer_model, time_duration
import copy
from torch.autograd import Variable
from model.model_VGG_cifar import construct_VGG_cifar
from model.model_AlexNet_cifar import construct_AlexNet_cifar
from model.model_nin_cifar import construct_nin_cifar
from model.model_VGG_image import construct_VGG_image
from model.model_AlexNet_image import construct_AlexNet_image
from model.model_nin_image import construct_nin_image
from model.model_nin_emnist import construct_nin_emnist
from model.model_AlexNet_emnist import construct_AlexNet_emnist
from model.model_VGG_emnist import construct_VGG_emnist
from util.utils import printer
import math
import numpy.ma as ma
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

parser = argparse.ArgumentParser(description='PyTorch MNIST SVM')
parser.add_argument('--device_num', type=int, default=1, metavar='N',
                        help='number of working devices ')
parser.add_argument('--edge_number', type=int, default=1, metavar='N',
                        help='edge server')
parser.add_argument('--model_type', type=str, default='NIN', metavar='N',          #NIN,AlexNet,VGG
                        help='model type')
parser.add_argument('--dataset_type', type=str, default='cifar10', metavar='N',  #cifar10,cifar100,image,emnist
                        help='dataset type')
args = parser.parse_args()

if True:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)
device_gpu = torch.device("cuda" if True else "cpu")
   
lr = 0.01
device_num = args.device_num
edge_num = args.edge_number
model_length = 0
delay_gap = 10
epoch_max = 500
acc_count = []
criterion = nn.NLLLoss()

listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#listening_sock.bind(('localhost', 50010))
listening_sock.bind(('172.16.50.22', 50010))

listening_sock1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#listening_sock1.bind(('localhost', 50002))
listening_sock1.bind(('172.16.50.22', 50011))

edge_sock_all = []

while len(edge_sock_all) < edge_num:
    listening_sock1.listen(edge_num)
    print("Waiting for incoming connections...")
    (client_sock, (ip, port)) = listening_sock1.accept()
    print('Got connection from ', (ip,port))
    print(client_sock)
    edge_sock_all.append(client_sock)

device_sock_all = [None]*device_num
#connect to device
for i in range(device_num):
    listening_sock.listen(device_num)
    print("Waiting for incoming connections...")
    (client_sock, (ip, port)) = listening_sock.accept()
    msg = recv_msg(client_sock)
    print('Got connection from node '+ str(msg[1]))
    print(client_sock)
    device_sock_all[msg[1]] = client_sock





#test the accuracy of model after aggregation
def test(models, dataloader, dataset_name, epoch, start_time):
    for model in models:
        model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            x=data.to(device_gpu)
            for i in range(0,len(models)):
                y = models[i](x)
                if i<len(models)-1:
                    x = y
                else:
                    loss += criterion(y, target)
                    pred = y.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
    end_time = time.time()
    a,b = time_duration(start_time, end_time)
    printer("Epoch {} Duration {}s {}ms Testing loss: {}".format(epoch,a,b,loss/len(dataloader)))
    printer("{}: Accuracy {}/{} ({:.0f}%)".format(dataset_name, 
                                                correct,
                                                len(dataloader.dataset),
                                                100. * correct / len(dataloader.dataset)))
    acc_count.append(correct*1.0/len(dataloader.dataset))

if args.dataset_type == 'cifar100':
    print("cifar100")
    transform = transforms.Compose([ 
                               #     transforms.RandomCrop(32, padding=4),
                                #    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                ])
    testset = datasets.ImageFolder('/data/zywang/Dataset/test_cifar100', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

elif args.dataset_type == 'cifar10':
    transform = transforms.Compose([ 
                                #    transforms.RandomCrop(32, padding=4),
                                #    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                ])
    testset = datasets.ImageFolder('/data/zywang/Dataset/cifar10/test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

elif args.dataset_type == 'emnist':
    transform = transforms.Compose([
                           transforms.Resize(28),
                           #transforms.CenterCrop(227),
                           transforms.Grayscale(1),
                           transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
          ])
    testset = datasets.ImageFolder('/data/zywang/Dataset/emnist/byclass_test', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)

elif args.dataset_type == 'image' and args.model_type != "AlexNet":
    transform = transforms.Compose([
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    testset = datasets.ImageFolder('/data/zywang/Dataset/IMAGE10/test', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)

elif args.dataset_type == 'image' and args.model_type == "AlexNet":
    transform = transforms.Compose([  transforms.Scale((227,227)),
                               #   transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                              ])
    testset = datasets.ImageFolder('/data/zywang/Dataset/IMAGE10/test', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)

#get the information about edge and device
def Get_commp_comm_mem_of_device_edge(str,epoch):
    compute_device = []
    communication_device = []
    memory_device = []
    data_size = []
    for i in range(device_num):
        msg = recv_msg(device_sock_all[i],"CLIENT_TO_SERVER") #compute, communication, memory,data_size 0-device1,1-device2...
        compute_device.append(msg[1])
        communication_device.append(msg[2])
        memory_device.append(msg[3])
        data_size.append(msg[4])
    compute_edge = []
    communication_edge = []
    memory_edge = []
    for i in range(edge_num):
        msg = recv_msg(edge_sock_all[i],"CLIENT_TO_SERVER") #compute, communication, memory
        compute_edge.append(msg[1])
        communication_edge.append(msg[2])
        memory_edge.append(msg[3])
    if str == 'NIN' and args.dataset_type == "image":
        model_transmission = [70.90,70.90,70.90,17.09,45.56,45.56,45.56,10.56,15.84,15.84,15.84,3.38,9.00,9.00,0.09,0]
        model_size = [1.0708,0.288,0.28,0,18.76,2.0170,2.0170,0,27.02,4.524,4.524,0,108.06,32.0639,0.3129,0]
        layer_memory = 2*(model_size + model_transmission)
    elif str == 'VGG' and args.dataset_type == "image":
        model_transmission = [784,784,196,392,392,98,196,196,196,49,98,98,98,24.5,24.5,24.5,24.5,6.13,1,1,0]
        model_size = [0.008,0.14,0,0.28,0.56,0,1.13,2.25,2.25,0,4.5,9.01,9.01,0,9.01,9.01,9.01,0,392.02,64.02,15.63]
        layer_memory = 2*(model_size + model_transmission)
    elif str == 'AlexNet' and args.dataset_type == "image":
        model_transmission = [70.9,17.09,45.56,10.56,15.84,15.84,10.56,2.25,1,1,0.24]
        model_size = [0.133,0,2.345,0,3.377,5.054,3.377,0,144.016,64.016,15.629,0]
        layer_memory = 2*(model_size + model_transmission)
    elif str == 'NIN' and args.dataset_type != "image":
        if args.dataset_type == 'emnist':
            model_transmission = [36.75,30.6,18.38,4.59,9.19,9.19,9.19,2.3,2.3,2.3,0.74,0]
            model_size = [0.056,0.11,0.05,0,1.759,0.142,0.1422,0,1.267,0.1422,0.007,0]
        else:
            model_transmission = [48.0,40.0,24.0,6.0,12.0,12.0,12.0,3.0,3.0,3.0,0.16,0]
            model_size = [0.056,0.11,0.05,0,1.759,0.142,0.1422,0,1.267,0.1422,0.007,0]
        layer_memory = 2*(model_size + model_transmission)
    elif str == 'VGG' and args.dataset_type != "image":
        model_transmission = [16.0,16.0,4.0,8.0,8.0,2.0,4.0,4.0,4.0,1.0,2.0,2.0,2.0,0.5,0.5,0.5,0.5,0.13,1.00,1.0,0.0]
        model_size = [0.133,0,2.345,0,3.377,5.054,3.377,0,144.016,64.016,15.629,0,0,0,0,0,0,0,0,0]
        layer_memory = 2*(model_size + model_transmission)
    elif str == 'AlexNet' and args.dataset_type != "image":
        model_transmission = [4.0,1.0,3.0,0.75,1.5,1,1,0.25,1,1,0.002]
        model_size = [0.133,0,2.345,0,3.377,5.054,3.377,0,144.016,64.016,15.629,0]
        layer_memory = 2*(model_size + model_transmission)
    partiiton_way, offloading_descision = partition_algorithm(compute_device, communication_device, memory_device, data_size,
          compute_edge, communication_edge, memory_edge, model_transmission, layer_memory,epoch)
    return partiiton_way, offloading_descision


def sdp_single(compute_device, communication_device_edge, compute_edge, model_transmission, succ, pred, memory_device, memory_edge, memory_layer, max_edge):              #single device and single edge
    alpha = []
    beta = []
    rank = [None]*len(compute_device[0])
    L = len(compute_device[0])
    partition_way = [None]*L
    TST = [None]*L
    TFT = [None]*L
    flag = False #device and edge have not enough memory
    if max_edge != None and sum(compute_device[0])+sum(compute_device[1]) < max_edge and sum(memory_layer)<memory_device:
        for i in range(L):
            partition_way[i] = 0
        memory_device -= sum(memory_layer)
     #   print("hello")
    else:
        for i in range(len(compute_device[0])):
            alpha.append((compute_device[0][i]+compute_device[1][i]+compute_edge[0][i]+compute_edge[1][i])/2)
            beta.append(model_transmission[i]/communication_device_edge*1000)
        rank[len(compute_device[0])-1] = beta[len(compute_device[0])-1]
        for i in range(L-2,-1,-1):
            rank[i] = alpha[i] + rank[i+1] + beta[i]
        for i in range(L):
            if i==0:
                partition_way[i] = 0
                TST[i] = 0
                TFT[i] = compute_device[0][0] + compute_device[1][0]
                memory_device = memory_device - memory_layer[0]
            else:
                if partition_way[i-1] == 0:
                    if TFT[i-1]+ 0 + compute_device[0][i] + compute_device[1][i] <= TFT[i-1] + 2*beta[i-1] + compute_edge[0][i] + compute_edge[1][i]:        
                        if memory_device-memory_layer[i]>0:
                            partition_way[i] = 0
                            memory_device = memory_device - memory_layer[i]
                            TST[i] = TFT[i-1]
                            TFT[i] = TST[i] + compute_device[0][i] + compute_device[1][i]
                        elif memory_edge - memory_layer[i]>0:   
                            partition_way[i] = 1
                            memory_edge = memory_edge - memory_layer[i]
                            TST[i] = TFT[i-1] + 2*beta[i-1]
                            TFT[i] = TST[i] + compute_edge[1][i]+compute_edge[0][i]
                        else:
                            flag = True
                            break
                    elif TFT[i-1]+ 0 + compute_device[0][i] + compute_device[1][i] > TFT[i-1] + 2*beta[i-1] + compute_edge[0][i] + compute_edge[1][i]:                                                                               
                        if memory_edge-memory_layer[i]>0:
                            partition_way[i] = 1
                            memory_edge = memory_edge - memory_layer[i]
                            TST[i] = TFT[i-1] + 2*beta[i-1]
                            TFT[i] = TST[i] + compute_edge[1][i]+compute_edge[0][i]
                        elif memory_device - memory_layer[i]>0:   
                            partition_way[i] = 0
                            memory_device = memory_device - memory_layer[i]
                            TST[i] = TFT[i-1]
                            TFT[i] = TST[i] + compute_device[0][i] + compute_device[1][i]
                        else:
                            flag = True
                            break
                        
                else:
                    if TFT[i-1]+ 0 + compute_edge[0][i] + compute_edge[1][i] <= TFT[i-1] + 2*beta[i-1] + compute_device[0][i] + compute_device[1][i]:                                    
                        if memory_edge-memory_layer[i]>0:
                            partition_way[i] = 1
                            memory_edge = memory_edge - memory_layer[i]
                            TST[i] = TFT[i-1] 
                            TFT[i] = TST[i] + compute_edge[1][i]+compute_edge[0][i]
                        elif memory_device - memory_layer[i]>0:   
                            partition_way[i] = 0
                            memory_device = memory_device - memory_layer[i]
                            TST[i] = TFT[i-1] + 2*beta[i-1]
                            TFT[i] = TST[i] + compute_device[0][i] + compute_device[1][i]
                        else:
                            flag = True
                            break
                    elif TFT[i-1]+ 0 + compute_edge[0][i] + compute_edge[1][i] > TFT[i-1] + 2*beta[i-1] + compute_device[0][i] + compute_device[1][i]:                                   
                        if memory_device-memory_layer[i]>0:
                            partition_way[i] = 0
                            memory_device = memory_device - memory_layer[i]
                            TST[i] = TFT[i-1] + 2*beta[i-1]
                            TFT[i] = TST[i] + compute_device[1][i]+compute_device[0][i]
                        elif memory_edge-memory_layer[i]>0:
                            partition_way[i] = 1
                            memory_edge = memory_edge - memory_layer[i]
                            TST[i] = TFT[i-1] 
                            TFT[i] = TST[i] + compute_edge[1][i]+compute_edge[0][i]
                        else:
                            flag = True
                            break
  #  print("sdp_single...............................")
    #print(alpha,beta,rank,L,partition_way, TST, TFT) 
    return partition_way, memory_edge, memory_device, flag, TFT[L-1]

def DAG_get(model):
    if model == 'Alexnet':
        DAG = [[0,1,0,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,0,1,0,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0,0],
                [0,0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,0,0,0,0]]
        succ = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[None]]
        pred = [[None],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
    else:
        pass
    return DAG, succ, pred


def sdp_offloading(data_size, compute_device, communication_device, memory_device, compute_edge, memory_edge, model_transmission, memory_layer ):
    N = len(compute_device)
    M = len(compute_edge)
    partition_way = [None]*N
    offloading_descision = [None]*N
    lamda = [None]*N
    if True:
        DAG, succ, pred = DAG_get('Alexnet')
    offloading_edge_side = []
    for i in range(M):
        offloading_edge_side.append([]) 
    sum_compute = 0
    for i in range(N):
        sum_compute+=sum(compute_device[i][0])+sum(compute_device[i][1])
    #ordering
    for i in range(N):
        lamda[i] = (data_size[i]/sum(data_size))/((sum(compute_device[i][0])+sum(compute_device[i][1]))/sum_compute+memory_device[i]/sum(memory_device))
   # print(lamda)
    #offloading and partition
    lamda_high = lamda
    for i in range(N):
        high = lamda_high.index(max(lamda_high))
        lamda_high[high] = -1
        TFT = 10000000
        memory_edge_high = 0
        memory_device_high = 0
        offloading = -1
        partition_way_high = []
        for j in range(M):
            pred = []
            succ = []
         #   print(M,compute_edge)
            partition_way_d, memory_edge_d, memory_device_d, flag, TFT_d = sdp_single(compute_device[high], communication_device[high][j], compute_edge[j], 
                        model_transmission, succ, pred, memory_device[high], memory_edge[j], memory_layer,None)
            if TFT_d < TFT and flag == False:
                partition_way_high = partition_way_d
                memory_device_high = memory_device_d
                memory_edge_high = memory_edge_d
                offloading = j
                TFT = TFT_d
        partition_way[high] = partition_way_high
        memory_device[high] = memory_device_high
        memory_edge[offloading] = memory_edge_high
        offloading_descision[high] = offloading
        if offloading_edge_side[offloading] == None:
            offloading_edge_side[offloading].append([])
        offloading_edge_side[offloading].append(high)

    #Strategy adjustment phase
    offloading_delay = []
    for i in range(M):
        offloading_delay.append([])
        for j in range(len(offloading_edge_side[i])):
            offloading_delay[i].append(delay_count(communication_device[offloading_edge_side[i][j]][i],compute_device[offloading_edge_side[i][j]],
                  compute_edge[i],model_transmission,partition_way[offloading_edge_side[i][j]]))
    while True:
        max_edge = []
        for i in range(M):
            max_edge.append(max_training_delay(offloading_delay[i],partition_way,offloading_edge_side[i]))
        edge_label = max_edge.index(max(max_edge))
        max_dealy = max(max_edge)
      #  print(max_dealy)
        device_label = offloading_edge_side[edge_label][0]
        location = 0
        for i in range(len(offloading_edge_side[edge_label])):
            if lamda[offloading_edge_side[edge_label][i]]< lamda[device_label] and 1 in partition_way[offloading_edge_side[edge_label][i]]:
                device_label = offloading_edge_side[edge_label][i]
                location = i
        a = offloading_delay[edge_label][location]
        offloading_delay[edge_label].pop(location)
        offloading_edge_side[edge_label].pop(location)
        for  i in range(len(partition_way[device_label])):
            if partition_way[device_label][i]==0:
                memory_device[device_label]+=memory_layer[i]
            else:
                memory_edge[edge_label]+=memory_layer[i]
        high = device_label
     #   print(high)
        partition_way_high = partition_way[high]
        flag1 =False
        for j in range(M):
            if j != edge_label :
                pred = []
                succ = []
                partition_way_d, memory_edge_d, memory_device_d, flag, TFT_d = sdp_single(compute_device[high], communication_device[high][j], compute_edge[j], 
                            model_transmission, succ, pred, memory_device[high], memory_edge[j], memory_layer, max_edge[j])   
                if flag == False and flag1 == False: 
                   # print("world")
                    offloading_delay[j].append(delay_count(communication_device[device_label][j],compute_device[device_label],
                        compute_edge[j],model_transmission,partition_way_d))  
                    partition_way[high] = partition_way_d
                    offloading_edge_side[j].append(device_label)
                    training_dealy = max_training_delay(offloading_delay[j],partition_way,offloading_edge_side[j])
                  #  print(partition_way_d,max_edge[j],offloading_delay[j],training_dealy,max_dealy)
                    if training_dealy + delay_gap < max_dealy:
                      #  print("worldsize")

                     #   print(training_dealy,max_dealy)
                        flag1 = True
                       # partition_way[high] = partition_way_d
                        memory_device[high] = memory_device_d
                        memory_edge[j] = memory_edge_d
                        offloading_descision[high] = j
                        offloading_edge_side[j].append(high)
                      #  offloading_delay[edge_label].remove(device_label)
                    else:
                        offloading_delay[j].pop()
                        offloading_edge_side[j].pop()
                        partition_way[device_label] = partition_way_high
            
        if flag1 == False:
            pred = []
            succ = []
            partition_way[high],memory_edge[edge_label],memory_device[high], flag, TFT_d = sdp_single(compute_device[high], communication_device[high][edge_label], compute_edge[edge_label], 
                        model_transmission, succ, pred, memory_device[high], memory_edge[edge_label], memory_layer, None)
            offloading_descision[high] = edge_label
           # offloading_delay[edge_label].append(a)
            offloading_edge_side[edge_label].append(high)
            break

    return offloading_descision, partition_way

def max_training_delay(offloading_delay,partition_way,offloading_edge_side):
    latency = 0
    for k in range(1):
        for i in range(len(offloading_delay)-1):
                 #   print(partition_way,offloading_edge_side,len(offloading_delay))
            latency += time_comparision(offloading_delay[i],offloading_delay[i+1],partition_way[offloading_edge_side[i]],
                                  partition_way[offloading_edge_side[i+1]])
    max_delay = 0
    for i in range(len(offloading_delay)):
        if offloading_delay[i][3][len(offloading_delay[i][3])-1] + latency>max_delay:
            max_delay = offloading_delay[i][3][len(offloading_delay[i][3])-1] + latency
    return max_delay

def delay_count(B, device_compute, edge_compute, data_tramission, partition_way):
    B = B/1000
    delay = []
    for i in range(4):
        delay.append([]) #0 forward communication #1 forward computing #2 backward communication #3 backward computing
    for i in range(len(partition_way)):    
        if i==0 and partition_way[i]!=0:
            delay[0].append(data_tramission[0]/B) 
        elif i==0 and partition_way[i]==0:
            delay[0].append(0)
        if i>0 and partition_way[i]==partition_way[i-1]:
            delay[0].append(delay[1][len(delay[1])-1])
        elif i>0 and partition_way[i]!=partition_way[i-1]:
            delay[0].append(delay[1][len(delay[1])-1]+ data_tramission[i-1]/B)
        if partition_way[i]==0:
            delay[1].append( device_compute[0][i] + delay[0][len(delay[0])-1])
        else:
            delay[1].append( edge_compute[0][i] + delay[0][len(delay[0])-1])

    for i in range(len(partition_way)-1,-1,-1):
        if i==model_length-1:
            delay[2].append(delay[1][len(delay[1])-1])
        if i<model_length-1 and partition_way[i]==partition_way[i+1]:
            delay[2].append(delay[3][len(delay[3])-1])
        elif i<model_length-1 and partition_way[i]!=partition_way[i+1]:
            delay[2].append(delay[3][len(delay[3])-1]+ data_tramission[i]/B)
        if partition_way[i]==0:
            delay[3].append( device_compute[1][i] + delay[2][len(delay[2])-1])
        else:
            delay[3].append( edge_compute[1][i] + delay[2][len(delay[2])-1]) 
    return delay


def time_update(latency, id0, id1, array):
    if id0==1:
        for j in range(id1, len(array[1])):
            array[1][j]+=latency
            array[0][j]+=latency
        array[2][0]=array[1][15]
        for j in range(len(array[1])):
            array[3][j]+=latency
            array[2][j]+=latency
    if id0==3:
        for j in range(id1, len(array[1])):
            array[3][j]+=latency
            array[2][j]+=latency
    return array


def time_comparision(array1, array2, partition_way1, partition_way2):
    latency = 0
    for i in range(len(array1[1])):
        for j in range(len(array1[1])):
            if array1[0][i]<=array2[0][j] and array2[0][j]<array1[1][i] and partition_way1[i]==1 and partition_way2[j]==1:
                latency += array1[1][i]-array2[0][j]
             #   array2 = time_update(latency, 1, j, array2)
            if array1[0][i]<=array2[2][j] and array2[2][j]<array1[1][i] and partition_way1[i]==1 and partition_way2[len(partition_way2)-1-j]==1:
                latency += array1[1][i]-array2[2][j]
              #  array2 = time_update(latency, 3, j, array2)
    for i in range(len(array1[1])):
        for j in range(len(array2[1])):
            if array1[2][i]<=array2[0][j] and array2[0][j]<array1[3][i] and partition_way1[len(partition_way2)-1-j]==1 and partition_way2[j]==1:
                latency += array1[3][i]-array2[0][j]
              #  array2 = time_update(latency, 1, j, array2)
            if array1[2][i]<=array2[2][j] and array2[2][j]<array1[3][i] and partition_way1[len(partition_way2)-1-j]==1 and partition_way2[len(partition_way2)-1-j]==1:
                latency += array1[3][i]-array2[2][j]
              #  array2 = time_update(latency, 3, j, array2)
    return latency



#partition algorithm of the paper
def partition_algorithm(compute_device, communication_device, memory_device, data_size, compute_edge, communication_edge, 
          memory_edge, model_transmission, layer_memory,epoch ):
   # if epoch == 0 or epoch >4:
  #  if True:
    if epoch >4:
        partition_way = []
        offloading_descision = []
        for i in range(device_num):
            partition_way.append([])
            partition_way[i].append(0)
            for j in range(model_length-1):
                partition_way[i].append(0)
         #   pooling_layer = [2,5,9,13,17]
         #   pooling_layer = [1,3,7]
            # random.shuffle(pooling_layer)
            # a = pooling_layer[0]
            # for j in range(1, a+1):
            #     partition_way[i].append(0)
            # for j in range(a+1,model_length):
            #     partition_way[i].append(1)
            
              #  partition_way[i].append(random.randint(0,1))
        for i in range(device_num):
            offloading_descision.append(i % edge_num+1)
    elif epoch == 2:
        offloading_descision, partition_way = sdp_offloading(data_size, compute_device, communication_device, 
                    memory_device, compute_edge, memory_edge, model_transmission, layer_memory )
        for i in range(device_num):
            offloading_descision[i] = i % edge_num + 1
    elif epoch ==0:
 #   else:
        partition_way = []
        offloading_descision = []
        for i in range(device_num):
            partition_way.append([])
            partition_way[i].append(0)
            
           # pooling_layer = [2,5,9,13,17]
         #   pooling_layer = [1,3,7]
          #  pooling_layer = [3]
            pooling_layer = [17]
            random.shuffle(pooling_layer)
            a = pooling_layer[0]
            for j in range(1, a+1):
                partition_way[i].append(0)
            for j in range(a+1,model_length):
                partition_way[i].append(1)
        for i in range(device_num):
            offloading_descision.append(i % edge_num+1)      

    elif epoch ==1:
 #   else:
        partition_way = []
        offloading_descision = []
        for i in range(device_num):
            partition_way.append([])
            partition_way[i].append(0)
            
            pooling_layer = [2,5,9,13,17]
          #  pooling_layer = [1,3,7]
          #  pooling_layer = [3]
          #  pooling_layer = [7]
            random.shuffle(pooling_layer)
            a = pooling_layer[0]
            for j in range(1, a+1):
                partition_way[i].append(0)
            for j in range(a+1,model_length):
                partition_way[i].append(1)
        for i in range(device_num):
            offloading_descision.append(i % edge_num+1)      

    return partition_way, offloading_descision


def send_msg_to_device_edge(sock_adr, msg):
    send_msg(sock_adr, msg)


#cionstruct the part of model for each node
def part_model_construct(partition_way, models):
    node_models = [None]*len(models)
    for i in range(len(models)):
        if partition_way[i] == 0:
            node_models[i] = copy.deepcopy(models[i])
    return node_models


#initation the model and send to devices and edges
def model_send_with_partition(partiiton_way,offloading_descision,models):
    send_device=[]
    for i in range(device_num):
      #  model_device = part_model_construct(partiiton_way[i], models)
        msg = ['SERVER_TO_CLIENT', partiiton_way[i], offloading_descision]
        send_device_msg = threading.Thread(target=send_msg_to_device_edge, args=(device_sock_all[i], msg))
        send_device_msg.start()

       # send_msg(device_sock_all[i],msg)

    for i in range(edge_num):
        msg = ['SERVER_TO_CLIENT', models ,partition_way, offloading_descision, time.time()]
        #send_msg(edge_sock_all[i], msg )
        send_edge_msg = threading.Thread(target=send_msg_to_device_edge, args=(edge_sock_all[i], msg))
        send_edge_msg.start()


#the algorithm stops when accuauracy of changed less than 2% in 10 epochs 
def train_stop():
    if len(acc_count)<11:
        return False
    max_acc = max(acc_count[len(acc_count)-10:len(acc_count)])
    min_acc = min(acc_count[len(acc_count)-10:len(acc_count)])
    if max_acc-min_acc <=0.001:
        return True
    else:
        return False


def rev_msg_edge(sock,epoch,edge_id,offloading_descision):
    global rec_models
    global rec_time
    msg = recv_msg(sock,"CLIENT_TO_SERVER")
  #  models = copy.deepcopy(scale_model(msg[1],offloading_descision.count(edge_id)/len(offloading_descision)))
    if offloading_descision.count(edge_id)!=0:
        rec_models.append(scale_model(msg[1],float(offloading_descision.count(edge_id)/len(offloading_descision))))
        rec_time[epoch].append(time.time()-msg[2]+msg[3])
    print(msg[3], offloading_descision.count(edge_id)/len(offloading_descision))
 #   rec_time[epoch].append(time.time()-msg[2]+msg[3])

rec_models = []
rec_time = []
communication_cost=0
#models, optimizers = construct_VGG([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
#models, optimizers = construct_AlexNet([0,0,0,0,0,0,0,0,0,0,0],lr)
if args.dataset_type == "image":
    if args.model_type == "NIN":
        models, optimizers = construct_nin_image([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 16
    elif args.model_type == "AlexNet":
        models, optimizers = construct_AlexNet_image([0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 11
    elif args.model_type == "VGG":
        models, optimizers = construct_VGG_image([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 21
elif args.dataset_type == 'emnist':
    if args.model_type == "NIN":
        models, optimizers = construct_nin_emnist([0,0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 12
    elif args.model_type == "AlexNet":
        models, optimizers = construct_AlexNet_emnist([0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 11
    elif args.model_type == "VGG":
        models, optimizers = construct_VGG_emnist([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 21
else:
    if args.model_type == "NIN":
        models, optimizers = construct_nin_cifar([0,0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 12
    elif args.model_type == "AlexNet":
        models, optimizers = construct_AlexNet_cifar([0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 11
    elif args.model_type == "VGG":
        models, optimizers = construct_VGG_cifar([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 21


for model in models:
    model_size = 0
    count = 0
    if model!=None:
        for para in model.parameters():
            model_size+=sys.getsizeof(para.storage())/(1024*1024)
        print("layer " +str(count) + "model size " +str(model_size)+"MB")
        count+=1
start_time = time.time()
for epoch in range(epoch_max):
    rec_time.append([])
    print(acc_count)
    partition_way, offloading_descision = Get_commp_comm_mem_of_device_edge(args.model_type,epoch)
    printer("partition_way_and_offloading_descision {},{} ".format(partition_way,offloading_descision))
    model_send_with_partition(partition_way, offloading_descision, models)
  #  print("epoch"+str(epoch)+'before update'+'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
  #  for model in models:
   #     for para in model.parameters():
        #    printer_model(para)
    rev_msg_d = []
    for i in range(edge_num):
       # msg = recv_msg(device_sock_all[i],"CLIENT_TO_SERVER") #get the parameter [0,weight]
        print("rec models")
        rev_msg_d.append(threading.Thread(target = rev_msg_edge, args = (edge_sock_all[i],epoch, i+1,offloading_descision)))
        rev_msg_d[i].start()
    for i in range(edge_num):
        rev_msg_d[i].join()
    for i in range(1,len(rec_models)):
        rec_models[0] = copy.deepcopy(add_model(rec_models[0], rec_models[i]))
    models = copy.deepcopy(rec_models[0])
    rec_models.clear()
    test(models, testloader, "Test set", epoch, start_time)
  #  print(rec_time)
    for i in range(len(rec_time[epoch])):
        communication_cost += rec_time[epoch][i]/len(rec_time[epoch])
    printer("Model_distribution_and_collection_time {} ".format(communication_cost))
  #  print("epoch"+str(epoch)+'before update'+'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#    for model in models:
      #  for para in model.parameters():
         #   printer_model(para)
    if train_stop():
        break

print("The traing process is over")


 
#tensor([ 0.0034, -0.0065, -0.0107, -0.0122, -0.0101, -0.0042, -0.0142, -0.0101,
    #     0.0074, -0.0019], requires_grad=True)
#tensor([ 0.0031, -0.0064, -0.0105, -0.0122, -0.0097, -0.0043, -0.0140, -0.0100,
 #        0.0069, -0.0021], requires_grad=True)

