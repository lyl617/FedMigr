#!/usr/bin/env python
import socket
import time
import torch
import sys
import time
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np
import torchvision
import torch.nn.functional as F
import random
import threading
import os
import socket
import time
import struct
from util.utils import send_msg, recv_msg
import copy
import argparse
from torch.autograd import Variable
from model.model_VGG_cifar import construct_VGG_cifar
from model.model_AlexNet_cifar import construct_AlexNet_cifar
from model.model_nin_cifar import construct_nin_cifar
from model.model_VGG_image import construct_VGG_image
from model.model_AlexNet_image import construct_AlexNet_image
from model.model_nin_image import construct_nin_image
from util.utils import printer, partition_way_converse, start_forward_layer, start_backward_layer, time_printer, add_model ,scale_model,printer_model
import numpy as np
from util.utils import send_msg, recv_msg

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='PyTorch MNIST SVM')
parser.add_argument('--device_num', type=int, default=1, metavar='N',
                        help='number of working devices ')
parser.add_argument('--edge_id', type=int, default=1, metavar='N',
                        help='edge server')
parser.add_argument('--model_type', type=str, default='NIN', metavar='N',          #NIN,AlexNet,VGG
                        help='model type')
parser.add_argument('--dataset_type', type=str, default='cifar10', metavar='N',  #cifar10,cifar100,image
                        help='dataset type')
parser.add_argument('--edge_ip', type=str, default='192.168.0.101', metavar='N',  #cifar10,cifar100,image
                        help='dataset type')
args = parser.parse_args()


if False:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

device_gpu = torch.device("cuda" if False else "cpu")

device_num = args.device_num
start_time = time.time()
lr=0.01
criterion = nn.NLLLoss()
edge_label = 1  #402
receive_model = []

listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#listening_sock.bind(('192.168.0.101', 51010))
listening_sock.bind(('192.168.0.127', 51001))
client_sock_all=[]

#connect to the PS
sock = socket.socket()
sock.connect(('172.16.50.10', 50011))

device_sock_all=[]
while len(device_sock_all) < device_num:
    listening_sock.listen(device_num)
    print("Waiting for incoming connections...")
    (client_sock, (ip, port)) = listening_sock.accept()
    print('Got connection from ', (ip,port))
    print(client_sock)
    device_sock_all.append(client_sock)
print("-------------------------------------------------------")
#receive partition way and construct model

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


def test(models, dataloader, dataset_name, epoch):
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
    printer("Epoch {} Testing loss: {}".format(epoch,loss/len(dataloader)))
    printer("{}: Accuracy {}/{} ({:.0f}%)".format(dataset_name, 
                                                correct,
                                                len(dataloader.dataset),
                                                100. * correct / len(dataloader.dataset)))
    

'''
transform = transforms.Compose([  #transforms.Scale((224,224)),
                                  transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                              ])
#trainset = datasets.CIFAR10('/data/zywang/Dataset/cifar10', download=True, train=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = datasets.CIFAR10('/data/zywang/Dataset/cifar10', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)

transform = transforms.Compose([
                           transforms.Resize(299),
                           transforms.CenterCrop(227),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
          ])
testset = datasets.ImageFolder('/data/zywang/Dataset/IMAGE10/test', transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=True)
'''



def train_model(client_sock, id,start_forward,start_backward, partition_way,input,output,models,optimizers):
    id=0
    global receive_model
    #start training
    while True:
        msg = recv_msg(client_sock,'CLIENT_TO_SERVER')
    # print(msg)
        if msg[2] == 2: #label
         #   msg[5] = msg[5].detach().requires_grad_(True)
            labels = msg[5].to(device_gpu)

        if msg[2] == 0: #forward
            if msg[4]==start_forward:#initiation
                start_forward==-1
                for i in range(0, len(partition_way)):
                    if partition_way[i]==0:
                        models[i].train()  
            current_layer = msg[4]
         #   print(msg[5].type())
          #  msg[5] = msg[5].detach().requires_grad_()
          #  print(msg[5])
            input[current_layer] = msg[5].to(device_gpu)
            input[current_layer] = input[current_layer].detach().requires_grad_()
          #  print(input[current_layer])

            while current_layer+1<len(partition_way):

                start_time = time.time()
                if partition_way[current_layer]==0 and partition_way[current_layer+1]==0:
                    output[current_layer] = models[current_layer](input[current_layer])
                    input[current_layer+1] = output[current_layer].detach().requires_grad_()
                    current_layer+=1
                    end_time = time.time()
                  #  time_printer(start_time,end_time,input[current_layer],current_layer,1)

                elif partition_way[current_layer]==0 and partition_way[current_layer+1]==1:
                    output[current_layer] = models[current_layer](input[current_layer])
                    input[current_layer+1] = output[current_layer].detach().requires_grad_()
                    end_time = time.time()
                   # time_printer(start_time,end_time,input[current_layer],current_layer,1)
                    msg_send = ['SERVER_TO_CLIENT',input[current_layer+1]]
                    send_msg(client_sock, msg_send)
                    break
            if current_layer == len(partition_way)-1:#到最后一层，回传
                start_backward = -1
                start_time=time.time()
                output[current_layer] = models[current_layer](input[current_layer])
                loss = criterion(output[current_layer], labels)
                end_time=time.time()
               # time_printer(start_time,end_time,input[current_layer],current_layer,1)
                for i in range(0,len(partition_way)):
                    if optimizers[i] !=None:
                        optimizers[i].zero_grad()
                start_time = time.time()
                loss.backward()
                end_time = time.time()
              #  time_printer(start_time,end_time,input[n][current_layer],current_layer,0)
                while partition_way[current_layer-1] == 0:

                    start_time = time.time()
                    grad_in = input[current_layer].grad
                    output[current_layer-1].backward(grad_in)
                    end_time = time.time()
                 #   time_printer(start_time,end_time,input[current_layer].grad,current_layer,0)
                    current_layer -= 1
                grad_in = input[current_layer].grad
                #print(grad_in)
                msg_send = ['SERVER_TO_CLIENT',grad_in]
                send_msg(client_sock, msg_send)
                if 0 in partition_way[0:current_layer]:  #后面没有需要继续需要backward的时，更新参数
                    pass
                else:
                    for i in range(0,len(partition_way)):
                        if optimizers[i] !=None:
                            optimizers[i].step()

        if msg[2] == 1: #backward
            if msg[4] == start_backward:
                start_backward = -1
                for i in range(0,len(partition_way)):
                    if optimizers[i] !=None and partition_way[i]==0:
                        optimizers[i].zero_grad()
            current_layer = msg[4]
            grad_in = msg[5].to(device_gpu)
            output[current_layer].backward(grad_in)
            while partition_way[current_layer-1] == 0:
                start_time= time.time()
                grad_in = input[current_layer].grad
                output[current_layer-1].backward(grad_in)
                end_time= time.time()
             #   time_printer(start_time,end_time,input[current_layer].grad,current_layer,0)
                current_layer -= 1
            grad_in = input[current_layer].grad
            msg_send = ['SERVER_TO_CLIENT',grad_in]
            send_msg(client_sock, msg_send)   
            if 0 in partition_way[0:current_layer]:  #后面没有需要继续需要backward的时，更新参数
                    pass
            else:
                for i in range(0,len(partition_way)):
                    if optimizers[i] !=None:
                        optimizers[i].step() 

        if msg[2] == 4: #model aggregation
            print("received model from device")
            w = msg[5]
            for i in range(0,len(models)):
                if partition_way[i] == 1:
                    models[i] = copy.deepcopy(w[i].to(device_gpu))
         #   for model in models:
            #    for para in model.parameters():
              #      printer_model(para)
            receive_model.append(models)
         #   test(receive_model[0], testloader, "Test set",998)
           # test(models, testloader, "Test set", 999)

         #   for model in receive_model:
             #   for para in model.parameters():
                  #  printer_model(para)
          #  print("model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
          #  for model in models:
           #     for para in model.parameters():
             #       print(para)
        #    print("model1>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
          #  for model in  receive_model:
             #   for para in model.parameters():
                #    print(para)
            break
        
def part_model_construct(partition_way, models):
    node_models = [None]*len(models)
    for i in range(len(models)):
        if partition_way[i] == 0:
            node_models[i] = copy.deepcopy(models[i])
    return node_models

def send_msg_to_device_edge(sock_adr, msg):
    send_msg(sock_adr, msg)


#after connect to the PS
while True:
    compute_edge = []
    for i in range(2):
        compute_edge.append([])
        for j in range(model_length):
            compute_edge[i].append(0)
    communication_edge = []
    memory_edge = 10*1024
    msg = ['CLIENT_TO_SERVER',compute_edge,communication_edge,memory_edge]
    send_msg(sock,msg)
    msg = recv_msg(sock,'SERVER_TO_CLIENT')
    global_model = copy.deepcopy(msg[1])
  #  receive_model = scale_model(msg[1],0)
    global_partition = msg[2]
    offloading_descision = msg[3]
    model_distribution_time = time.time()-msg[4]

    id = -1
    client=[]
    partition_way=[]
    models=[]
    optimizers=[]
    input=[]
    output=[]
    start_forward=[]
    start_backward=[]
    for i in range(device_num):
        if edge_label == offloading_descision[i]:

            model_device = part_model_construct(global_partition[i], global_model)
            msg = ['SERVER_TO_CLIENT', model_device,time.time()]
           # send_msg(device_sock_all[i],msg)
            send_device_msg = threading.Thread(target=send_msg_to_device_edge, args=(device_sock_all[i], msg))
            send_device_msg.start()

            id += 1
            partition_way.append(global_partition[i])
            partition_way[id]= partition_way_converse(partition_way[id])
            models.append([None]*len(partition_way[id]))
            optimizers.append([None]*len(partition_way[id]))
            input.append([None]*len(partition_way[id]))
            output.append([None]*len(partition_way[id]))
            start_forward.append(start_forward_layer(partition_way[id]))
            start_backward.append(start_backward_layer(partition_way[id]))
          #  models[id], optimizers[id] = construct_AlexNet(partition_way[id],lr)
            if args.dataset_type == "image":
                if args.model_type == "NIN":
                    models[id], optimizers[id] = construct_nin_image([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
                elif args.model_type == "AlexNet":
                    models[id], optimizers[id] = construct_AlexNet_image([0,0,0,0,0,0,0,0,0,0,0],lr)
                elif args.model_type == "VGG":
                    models[id], optimizers[id] = construct_VGG_image([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
            else:
                if args.model_type == "NIN":
                    models[id], optimizers[id] = construct_nin_cifar([0,0,0,0,0,0,0,0,0,0,0,0],lr)
                elif args.model_type == "AlexNet":
                    models[id], optimizers[id] = construct_AlexNet_cifar([0,0,0,0,0,0,0,0,0,0,0],lr)
                elif args.model_type == "VGG":
                    models[id], optimizers[id] = construct_VGG_cifar([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
            print(partition_way[id])
            for j in range(0,len(global_model)):
                if partition_way[id][j] == 0:
                    models[id][j] = copy.deepcopy(global_model[j])
            for j in range(len(optimizers[id])):
                if optimizers[id][j]!=None and partition_way[id][j]==0:
                    optimizers[id][j]= optim.SGD(params = models[id][j].parameters(), lr = lr)
            #model in GPU
            client.append(threading.Thread(target=train_model, 
                        args=(device_sock_all[i], id,start_forward[id],start_backward[id], partition_way[id],input[id],output[id],models[id],optimizers[id])))
            client[id].start()
    for i in range(len(client)):
        client[i].join()
    for i in range(1,len(receive_model)):
        receive_model[0] = copy.deepcopy(add_model(receive_model[0], receive_model[i]))
    receive_model[0] = copy.deepcopy(scale_model(receive_model[0], 1.0/len(client)))
  #  print("model2>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  #  for model in  receive_model:
      #  for para in model.parameters():
         #   print(para)
    msg = ['CLIENT_TO_SERVER',receive_model[0],time.time(),model_distribution_time]
    print("send model")
    send_msg(sock, msg)
    receive_model.clear()
    del models
    del optimizers
    del client
            
