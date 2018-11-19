import gc
import sys
import time
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
torch.backends.cudnn.benchmark=True

import dataset
from models.AlexNet import *
from models.ResNet import *

def compute_err(data_loader, model, device):
    total = 0; top1_err = 0; top5_err = 0;
    for batch_num, (inputs, labels) in enumerate(data_loader, 1):
        inputs = inputs.to(device)
        labels = labels.to(device).view(-1,1)
        outputs = model(inputs)
        top1 = outputs.topk(1)[1]
        top5 = outputs.topk(5)[1]

        top1_err += (top1 != labels).sum()
        top5_err += labels.size()[0] - (top5 == labels).sum()

        total += labels.size()[0]
    top1_err_score = top1_err.float()/total
    top5_err_score = top5_err.float()/total
    return top1_err_score, top5_err_score

def run():
    t_start = time.time()
    # Parameters
    num_epochs = 10
    output_period = 100

    batch_size = 20
    learning_rate = 1e-2
    weight_decay = 0

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet_18()
    model = model.to(device)

    train_loader, val_loader = dataset.get_data_loaders(batch_size)
    num_train_batches = len(train_loader)

    criterion = nn.CrossEntropyLoss().to(device)
    # TODO: optimizer is currently unoptimized
    # there's a lot of room for improvement/different optimizers
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    epoch = 1
    while epoch <= num_epochs:
        t_start_epoch = time.time()
        running_loss = 0.0
        for param_group in optimizer.param_groups:
            print('Current learning rate: ' + str(param_group['lr']))
        model.train()

        for batch_num, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            if batch_num % output_period == 0:
                print('[%d:%.2f] loss: %.3f' % (
                    epoch, batch_num*1.0/num_train_batches,
                    running_loss/output_period
                    ))
                running_loss = 0.0
                gc.collect()

        gc.collect()
        # save after every epoch
        torch.save(model.state_dict(), "models/model.%d" % epoch)
        t_trained_epoch = time.time()

        # TODO: Calculate classification error and Top-5 Error
        # on training and validation datasets here
        with torch.no_grad():
            batch_size_eval = 100
            train_loader_eval, val_loader_eval = dataset.get_data_loaders(batch_size_eval)
            model.eval()
            print("Computing error on whole dataset...")
            train_top1_err, train_top5_err = compute_err(train_loader_eval, model, device)
            val_top1_err, val_top5_err = compute_err(val_loader_eval, model, device)
            print("[{epoch}:train] top1: {top1:.3f} top5: {top5:.3f}.".format(epoch=epoch, top1=train_top1_err, top5=train_top5_err))
            print("[{epoch}:validate] top1: {top1:.3f} top5: {top5:.3f}.".format(epoch=epoch, top1=val_top1_err, top5=val_top5_err))
            model.train()

        gc.collect()
        epoch += 1
        
        t_done_epoch = time.time()
        t_train_epoch = t_trained_epoch - t_start_epoch
        t_eval_epoch = t_done_epoch - t_trained_epoch
        t_total_epoch = t_done_epoch - t_start_epoch
        t_total_so_far = t_done_epoch - t_start
        print("Epoch took {:6.1f} sec total ({:6.1f} to train, {:6.1f} to eval)".format(t_total_epoch, t_train_epoch, t_eval_epoch))
        print("Taken {:10.1f} total so far.".format(t_total_so_far))


print('Starting training')
run()
print('Training terminated')
