# _*_ coding: utf-8 _*_
# This file is created by C. Zhang for personal use.
# @Time         : 18/08/2022 23:27
# @Author       : tl22089
# @File         : federated_adapt.py
# @Affiliation  : University of Bristol
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import datetime
import torch
import numpy as np
import random
import os
import copy
import time
import pickle
import h5py
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from rff.utils import get_dataset, average_weights
from rff.options import args_parser
from rff.mta import test_inference, LocalUpdate, DatasetSplit
from rff.models import MLP, CNN
from rff.resnet import ResNet18
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    args.type = 'metric'
    seed = 2022
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = 'cuda' if args.gpu else 'cpu'
    print(device)
    # define paths
    now = datetime.datetime.now()
    log_name = '../logs/Fed-metric-{:}-{:}-{:}-{:}-{:}-{:}-model-{:}-iid-{:}-bs-{:}/'.format(now.year, now.month,
                                                                                             now.day,
                                                                                             now.hour,
                                                                                             now.minute,
                                                                                             now.second,
                                                                                             args.model,
                                                                                             args.iid,
                                                                                             args.bs_classes)
    logger = SummaryWriter(log_name)
    # load datasets
    train_dataset, adapt_dataset, test_dataset, train_groups, adapt_groups = get_dataset(args)

    train_idx = [i for i in range(len(train_dataset))]
    train_ds = DatasetSplit(train_dataset, train_idx)

    adapt_idx = [i for i in range(len(adapt_dataset))]
    adapt_idx = np.random.choice(adapt_idx, int(len(adapt_dataset) * 0.01))
    adapt_ds = DatasetSplit(adapt_dataset, adapt_idx)

    test_idx = [i for i in range(len(test_dataset))]
    test_ds = DatasetSplit(test_dataset, test_idx)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural network
        global_model = CNN(args=args)
    elif args.model == 'resnet':
        global_model = ResNet18(args=args)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        global_model = MLP(args=args)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    if args.transfer:
        global_model.load_state_dict(torch.load('../save/fed_metric_{:}.pt'.format(args.model)))

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    train_bs_acc, train_radio_acc = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    distance = distances.LpDistance().to(device)
    reducer = reducers.ThresholdReducer(low=0).to(device)
    loss_func = losses.TripletMarginLoss(margin=args.margin, distance=distance, reducer=reducer).to(device)
    mine_func = miners.TripletMarginMiner(margin=args.margin, distance=distance, type_of_triplets='semihard').to(device)
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
    kw = {'loss_func': loss_func,
          'mine_func': mine_func, 'accuracy_calculator': accuracy_calculator}

    for epoch in tqdm(range(2)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=adapt_dataset,
                                      idxs=adapt_groups[idx], logger=logger, kw=kw)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        logger.add_scalar('train/loss', loss_avg, epoch)

        # # Calculate avg training accuracy over all users at every epoch
        # list_acc, list_loss, bs_acc, radio_acc = [], [], [], []
        # global_model.eval()
        # for c in range(args.num_users):
        #     local_model = LocalUpdate(args=args, dataset=train_dataset,
        #                               idxs=user_groups[c], logger=logger, kw=kw)
        #     acc = local_model.inference(model=global_model)
        #     list_acc.append(acc)
        # train_accuracy.append(sum(list_acc) / len(list_acc))
        # logger.add_scalar('train/acc', train_accuracy[-1], epoch)

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            # print('Train Accuracy: {:.2f}%\n'.format(100. * train_accuracy[-1]))

    # torch.save(global_model.state_dict(), '../save/fed_adapt_cnn.pt')
    # Test inference after completion of training
    test_acc = test_inference(train_set=adapt_ds,
                              test_set=test_ds,
                              model=global_model,
                              accuracy_calculator=accuracy_calculator)

    print(f' \n Results after {args.epochs} global rounds of training:')
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100. * test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/fed_metric_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_metric_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))
    #
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_metric_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))