# _*_ coding: utf-8 _*_
# This file is created by C. Zhang for personal use.
# @Time         : 18/08/2022 17:48
# @Author       : tl22089
# @File         : centralized_metric.py
# @Affiliation  : University of Bristol
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import numpy as np
import os
import torch
import random
import h5py
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rff.utils import get_dataset, AverageMeter
from rff.options import args_parser
from rff.update import DatasetSplit
from rff.models import MLP, CNN
from rff.resnet import ResNet18
from rff.mta import metric_classification
from sklearn import metrics
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def get_embed(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def test(train_set, test_set, model, accuracy_calculator):
    train_embed, train_labels = get_embed(train_set, model)
    test_embed, test_labels = get_embed(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embed, train_embed, test_labels, train_labels, False
    )
    print('acc:', accuracies)
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))


if __name__ == '__main__':
    args = args_parser()
    args.type = 'metric'
    seed = 2022
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = 'cuda' if args.gpu else 'cpu'
    # print(device)
    # define paths
    now = datetime.datetime.now()
    log_name = '../logs/Central-metric-{:}-{:}-{:}-{:}-{:}-{:}-model-{:}-iid-{:}-bs-{:}/'.format(now.year, now.month,
                                                                                                 now.day,
                                                                                                 now.hour,
                                                                                                 now.minute,
                                                                                                 now.second,
                                                                                                 args.model,
                                                                                                 args.iid,
                                                                                                 args.bs_classes)
    logger = SummaryWriter(log_name)
    # load datasets
    train_dataset, adapt_dataset, test_dataset, _, _ = get_dataset(args)
    n = len(train_dataset)
    train_dataset = train_dataset[np.random.choice(n, int(n / 5), replace=False)]

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

    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    train_idx = [i for i in range(len(train_dataset))]
    train_ds = DatasetSplit(train_dataset, train_idx)

    adapt_idx = [i for i in range(len(adapt_dataset))]
    adapt_ds = DatasetSplit(adapt_dataset, adapt_idx)

    test_idx = [i for i in range(len(test_dataset))]
    test_ds = DatasetSplit(test_dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    adapt_loader = DataLoader(adapt_ds, batch_size=256, shuffle=True)
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=args.margin, reducer=reducer)
    mine_func = miners.TripletMarginMiner(margin=args.margin, type_of_triplets='semihard')
    acc_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    train_loss = AverageMeter()
    train_acc = AverageMeter()
    test_loss = AverageMeter()
    test_acc = AverageMeter()

    epoch_loss = []
    history = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []
        global_model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings = global_model(data)
            indices_triple = mine_func(embeddings, labels)
            loss = loss_func(embeddings, labels, indices_triple)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss) / len(batch_loss)
        train_loss.update(loss_avg)

        ta, _, _ = metric_classification(global_model, train_ds, test_ds)
        test_acc.update(ta)

        history.append((epoch + 1, train_loss.val, train_loss.avg, test_acc.val, test_acc.avg))

    df_log = pd.DataFrame(history, columns=['epoch', 'train_loss_val', 'train_loss_avg',
                                            'test_acc_val', 'test_acc_avg'])

    global_model.eval()
    te_acc, te_pred, te_gt = metric_classification(global_model, train_ds, test_ds)
    tra_acc, tra_pred, tra_gt = metric_classification(global_model, train_ds, train_ds)

    fn = '../save/centralized_metric_{:}_{:}_{:}.h5'.format(args.signal, args.model, args.snr_high)
    mn = '../save/centralized_metric_{:}_{:}_{:}.pt'.format(args.signal, args.model, args.snr_high)

    torch.save(global_model.state_dict(), mn)

    fo = h5py.File(fn, 'w')
    fo.create_dataset('log', data=df_log.values)
    fo.create_dataset('train_pred', data=tra_pred)
    fo.create_dataset('train_label', data=tra_gt)
    fo.create_dataset('test_pred', data=te_pred)
    fo.create_dataset('test_label', data=te_gt)
    fo.close()
