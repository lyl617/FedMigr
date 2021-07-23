#!/bin/bash
# hyperparams

#
option=4
#
if [ $option -eq 1 ]; then #FashionMNIST CNN
visible_cuda='0'
batch_size=50
lr=0.001
min_lr=0.001
vm_num=10
local_iters=1
log_interval=100
pattern_idx=0 # 0-bias, 2-random
dataset_type='FashionMNIST' #lr 0.001-0.0001
model_type='CNN'
checkpoint_dir='./Checkpoint/Checkpoint_CENT_data_pattern_0_FashionMNIST/'
enable_lr_decay=1
epochs=2000
epoch_step=100
aggregate_interval=20
epoch_start_idxs=(0 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900)

elif [ $option -eq 2 ]; then #CIFAR10 CNN
visible_cuda='1'
batch_size=50
lr=0.01
min_lr=0.0001
vm_num=10
local_iters=1
log_interval=100
pattern_idx=0 # ['random', 'lowbias','midbias', 'highbias']
dataset_type='CIFAR10' #lr 0.01-0.0001
model_type='CNN'
checkpoint_dir='./Checkpoint/Checkpoint_CENT_data_pattern_0_CIFAR10/'
enable_lr_decay=1
epochs=2000
aggregate_interval=20
epoch_step=100
epoch_start_idxs=(0 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900)
#epoch_start_idxs=(0)

elif [ $option -eq 3 ]; then #CIFAR100 CNN
visible_cuda='2'
batch_size=50
lr=0.01
min_lr=0.0001
vm_num=10
local_iters=1
log_interval=100
pattern_idx=2 # ['random', 'lowbias','midbias', 'highbias']
dataset_type='CIFAR100' #lr 0.01-0.0001
model_type='CNN'
checkpoint_dir='./Checkpoint/Checkpoint_CENT_data_pattern_2_CIFAR100/'
enable_lr_decay=1
epochs=2000
aggregate_interval=20
epoch_step=100
epoch_start_idxs=(0 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900)

elif [ $option -eq 4 ]; then #MNIST LR
visible_cuda='2'
batch_size=50
lr=0.01
min_lr=0.001
vm_num=10
local_iters=1
log_interval=100
pattern_idx=0 # 0-bias, 2-random
dataset_type='MNIST' #lr 0.001-0.0001
model_type='LR'
checkpoint_dir='./Checkpoint/Checkpoint_CENT_data_pattern_0_MNIST/'
enable_lr_decay=1
aggregate_interval=20
epochs=2000
epoch_step=100
epoch_start_idxs=(0 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900)
fi

#----------------------------------------------

for epoch_start in ${epoch_start_idxs[*]}
do
echo "Total: $epochs, start $epoch_start, step $epoch_step"
if [ $enable_lr_decay -eq 1 ]; then
    python fl_model_central_update_gpu_sh.py --epochs $epochs --epoch_start $epoch_start --epoch_step $epoch_step \
                                             --batch_size $batch_size --lr $lr --min_lr $min_lr --vm_num $vm_num --local_iters $local_iters --log_interval $log_interval \
                                             --aggregate_interval $aggregate_interval --dataset_type $dataset_type --model_type $model_type --pattern_idx $pattern_idx --checkpoint_dir $checkpoint_dir \
                                             --seed $RANDOM --visible_cuda $visible_cuda --no_cuda --save_model --train_flag --enable_lr_decay
else
    python fl_model_central_update_gpu_sh.py --epochs $epochs --epoch_start $epoch_start --epoch_step $epoch_step \
                                             --batch_size $batch_size --lr $lr --min_lr $min_lr --vm_num $vm_num --local_iters $local_iters --log_interval $log_interval \
                                             --aggregate_interval $aggregate_interval --dataset_type $dataset_type --model_type $model_type --pattern_idx $pattern_idx --checkpoint_dir $checkpoint_dir \
                                             --seed $RANDOM --visible_cuda $visible_cuda --no_cuda --save_model --train_flag
fi
done