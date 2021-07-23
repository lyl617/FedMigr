__author__ = 'yang.xu'
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

from fl_utils import printer, time_since
import gc
import resource

def train(args, vm_models, device, vm_train_loaders, vm_test_loaders, vm_optimizers, param_server, epoch, start, fid):
    # param_server.clear_objects()
    for vm_idx in range(args.vm_num):
        vm_start = time.time()
        vm_models[vm_idx].train()
        for li_idx in range(args.local_iters):
            # <-- now it is a distributed dataset
            for batch_idx, (vm_data, vm_target) in enumerate(vm_train_loaders[vm_idx], 1):
                # print(data.location)
                
                if args.dataset_type == 'FashionMNIST' or args.dataset_type == 'MNIST':
                    if args.model_type == 'LR':
                        vm_data = vm_data.squeeze(1) 
                        vm_data = vm_data.view(-1, 28 * 28)
                    else:
                        # vm_data = vm_data.unsqueeze(1)  # <--for FashionMNIST
                        pass

                if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
                    if args.model_type == 'LSTM':
                        vm_data = vm_data.permute(0, 2, 3, 1)
                        vm_data = vm_data.contiguous().view(-1, 32, 32 * 3)                    
                    else:
                        # vm_data = vm_data.permute(0, 3, 1, 2) #<--for CIFAR10 & CIFAR100
                        pass
                if args.dataset_type == 'Sent140':
                    if args.model_type == 'GRU':
                        vm_data = vm_data.permute(1, 0, 2)                 
                    else:
                        pass

                vm_data, vm_target = vm_data.to(device), vm_target.to(device)
                # print('--[Debug] vm_data.shape ', vm_data.shape)
                # print('--[Debug] vm_target.shape ', vm_target.shape)
                #print('--[Debug] vm_data = ', vm_data.get())
                if args.model_type == 'GRU':
                    # initial hidden state 
                    num_layers=1
                    hidden_size=128
                    batch_size=args.batch_size
                    init_hidden_state = torch.zeros(num_layers, batch_size, hidden_size)
                    init_hidden_state = init_hidden_state.to(device)
                    init_hidden_state_ptr = init_hidden_state.send(vm_data.location)
                    #
                    vm_output = vm_models[vm_idx](vm_data, init_hidden_state_ptr)
                else:
                    vm_output = vm_models[vm_idx](vm_data)

                vm_optimizers[vm_idx].zero_grad()
                
                vm_loss = F.nll_loss(vm_output, vm_target)
                vm_loss.backward()
                vm_optimizers[vm_idx].step()
                # vm_data = vm_data.get()

                if batch_idx % args.log_interval == 0:
                    vm_loss = vm_loss.get().item()  # <-- NEW: get the loss back

                    printer('-->[{}] Train Epoch: {} Local Iter: {} vm: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        time_since(vm_start), epoch, li_idx, vm_idx, batch_idx * args.batch_size, 
                        len(vm_train_loaders[vm_idx]) * args.batch_size,
                        100. * batch_idx / len(vm_train_loaders[vm_idx]), vm_loss), fid)

                vm_data = vm_data.get()
                vm_target = vm_target.get()
                vm_output = vm_output.get()

                if not batch_idx % args.log_interval == 0:
                    vm_loss = vm_loss.get()
                
                del vm_data
                del vm_target
                del vm_output
                del vm_loss
                # break

        # if epoch == args.epochs:
        if args.enable_vm_test:
            printer('-->[{}] Test set: Epoch: {} vm: {}'.format(time_since(vm_start), epoch, vm_idx), fid)
            # <--test for each vm
            test(args, vm_models[vm_idx], device, vm_test_loaders[vm_idx], epoch, vm_start, fid)

        vm_models[vm_idx].move(param_server)
        # vm_models[vm_idx] = vm_models[vm_idx].get()
        # torch.cuda.empty_cache()
        gc.collect()

def test(args, model, device, test_loader, epoch, start, fid, total_bandwidth, result):
    model.eval()

    test_loss = 0.0
    test_accuracy = 0.0

    correct = 0

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            if args.dataset_type == 'FashionMNIST' or args.dataset_type == 'MNIST':
                if args.model_type == 'LR':
                    data = data.squeeze(1) 
                    data = data.view(-1, 28 * 28)
                else:
                    # vm_data = vm_data.unsqueeze(1)  # <--for FashionMNIST
                    pass

            if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
                if args.model_type == 'LSTM':
                    data = data.view(-1, 32, 32 * 3)                    
                else:
                    # vm_data = vm_data.permute(0, 3, 1, 2) #<--for CIFAR10 & CIFAR100
                    pass  

            if args.model_type == 'LSTM':
                hidden = model.initHidden(args.test_batch_size)
                hidden = hidden.send(data.location)
                for col_idx in range(32):
                    data_col = data[:, col_idx, :]
                    output, hidden = model(data_col, hidden)
            else:
                output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').get().item()
            # get the index of the max log-probability
            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().get().item()

            correct += batch_correct
            #print('--[Debug][in Test set] batch correct:', batch_correct)
            if not args.enable_vm_test:
                printer('--[Debug][in Test set] batch correct: {}'.format(batch_correct), fid)
            
            data =  data.get()
            target = target.get()
            output = output.get()
            pred = pred.get()
                
            del data
            del target
            del output
            del pred
            del batch_correct

    test_loss /= len(test_loader.federated_dataset)
    test_accuracy = np.float(1.0 * correct / len(test_loader.federated_dataset))

    if args.enable_vm_test:  
        printer('-->[{}] Test set: Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            time_since(start), epoch, test_loss, correct, len(test_loader.federated_dataset),
            100. * test_accuracy), fid)
    else:
        printer('\n[{}] Test set: Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            time_since(start), epoch, test_loss, correct, len(test_loader.federated_dataset),
            100. * test_accuracy), fid)
        printer('\n{} {} {:.4f} {:.4f} {:.4f}\n'.format(epoch, time_since(start), test_loss, test_accuracy, total_bandwidth), fid)
        printer('\n{} {} {:.4f} {:.4f} {:.4f}\n'.format(epoch, time_since(start), test_loss, test_accuracy, total_bandwidth), result)
    gc.collect()

    return test_loss, test_accuracy


def centralized_train(args, centralized_model, device, centralized_train_loader, centralized_test_loader, centralized_optimizer, epoch, start, fid):
    vm_start = time.time()
    centralized_model.train()
    # <-- now it is a distributed dataset
    for batch_idx, (vm_data, vm_target) in enumerate(centralized_train_loader, 1):
                # print(data.location)
                # print('refcount 1', sys.getrefcount(vm_data))
        if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
                # vm_data = vm_data.permute(0, 3, 1, 2) #<--for CIFAR10 & CIFAR100
            pass
        elif args.dataset_type == 'FashionMNIST':
            vm_data = vm_data.unsqueeze(1)  # <--for FashionMNIST

        vm_data, vm_target = vm_data.to(device), vm_target.to(device)

        centralized_optimizer.zero_grad()
        vm_output = centralized_model(vm_data)
        vm_loss = F.nll_loss(vm_output, vm_target)
        vm_loss.backward()
        centralized_optimizer.step()
        vm_data = vm_data.get()

        if batch_idx % args.log_interval == 0:
            vm_loss = vm_loss.get().data  # <-- NEW: get the loss back

            printer('-->[{}] (Centrailized) Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time_since(vm_start), epoch, batch_idx *
                args.batch_size, len(centralized_train_loader) * args.batch_size,
                100. * batch_idx / len(centralized_train_loader), vm_loss), fid)

        del vm_data
        del vm_target
        del vm_output
        del vm_loss

    # if epoch == args.epochs:
    if args.enable_vm_test:
        printer('-->[{}] (Centrailized) Test set: Epoch: {}'.format(time_since(vm_start), epoch), fid)
        # <--test for each vm
        test(args, centralized_model, device, centralized_test_loader, epoch, vm_start, fid)

    gc.collect()
