# _*_ coding: utf-8 _*_
# This file is created by C. Zhang for personal use.
# @Time         : 18/08/2022 16:07
# @Author       : tl22089
# @File         : centralized_baseline.py
# @Affiliation  : University of Bristol
import pandas as pd
from tqdm import tqdm
import datetime
import torch
import numpy as np
import random
import os
import h5py
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rff.utils import get_dataset, AverageMeter
from rff.options import args_parser
from rff.update import test_inference, DatasetSplit
from rff.models import MLP, DenseNet, CNN
from rff.resnet import ResNet18
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    args = args_parser()
    args.type = 'non-metric'
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
    log_name = '../logs/Central-baseline-{:}-{:}-{:}-{:}-{:}-{:}-model-{:}-iid-{:}-bs-{:}/'.format(now.year, now.month,
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

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural network
        global_model = CNN(args=args)
    elif args.model == 'resnet':
        global_model = ResNet18(args=args)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        global_model = MLP(args=args)
    elif args.model == 'densenet':
        global_model = DenseNet(args=args, depth=22)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    # global_model.load_state_dict(torch.load('../save/centralized.pt'))
    global_model.to(device)
    global_model.train()
    # print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    train_idx = [i for i in range(len(train_dataset))]
    trainloader = DataLoader(DatasetSplit(train_dataset, train_idx), batch_size=256, shuffle=True)
    loss_func = torch.nn.CrossEntropyLoss().to(device)

    train_loss = AverageMeter()
    train_acc = AverageMeter()
    test_loss = AverageMeter()
    test_acc = AverageMeter()

    history = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx, (data, labels) in enumerate(trainloader):
            data, labels = data.float().to(device), labels.long().to(device)
            optimizer.zero_grad()
            outputs = global_model(data)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss) / len(batch_loss)
        train_loss.update(loss_avg)
        # logger.add_scalar('train/loss', loss_avg, epoch)

        # testing
        ta, tl, _, _ = test_inference(args, global_model, test_dataset)
        test_loss.update(tl)
        test_acc.update(ta)

        history.append((epoch+1, train_loss.val, train_loss.avg, test_loss.val,
                        test_loss.avg, test_acc.val, test_acc.avg))

    df_log = pd.DataFrame(history, columns=['epoch', 'train_loss_val', 'train_loss_avg',
                                            'test_loss_val', 'test_loss_avg', 'train_acc_val',
                                            'train_acc_avg'])
    # testing
    te_acc, te_loss, te_pred, te_gt = test_inference(args, global_model, test_dataset)
    # print('Test on', len(test_dataset), 'samples')
    # print("|---- Test Accuracy on Test: {:.2f}%".format(100. * test_acc))

    # testing
    tra_acc, tra_loss, tra_pred, tra_gt = test_inference(args, global_model, train_dataset)
    # print('Test on', len(train_dataset), 'samples')
    # print("|---- Test Accuracy on Train: {:.2f}%".format(100. * test_acc))

    fn = '../save/centralized_base_{:}_{:}_{:}.h5'.format(args.signal, args.model, args.snr_high)
    mn = '../save/centralized_base_{:}_{:}_{:}.pt'.format(args.signal, args.model, args.snr_high)

    torch.save(global_model.state_dict(), mn)

    fo = h5py.File(fn, 'w')
    fo.create_dataset('log', data=df_log.values)
    fo.create_dataset('train_pred', data=tra_pred)
    fo.create_dataset('train_label', data=tra_gt)
    fo.create_dataset('test_pred', data=te_pred)
    fo.create_dataset('test_label', data=te_gt)
    fo.close()
