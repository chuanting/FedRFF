# _*_ coding: utf-8 _*_
# This file is created by C. Zhang for personal use.
# @Time         : 19/08/2022 21:13
# @Author       : tl22089
# @File         : metric_inference.py
# @Affiliation  : University of Bristol
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import numpy as np
import os
import torch
import random
import h5py
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rff.utils import get_dataset, AverageMeter
from rff.options import args_parser
from rff.update import DatasetSplit
from rff.models import MLP, CNN
from rff.resnet import ResNet18
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
    print(device)
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
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=True)
    distance = distances.LpDistance()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=args.margin, distance=distance, reducer=reducer)
    mine_func = miners.TripletMarginMiner(margin=args.margin, distance=distance, type_of_triplets='semihard')
    acc_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    epoch_loss = []

    global_model.load_state_dict(torch.load('../save/centralized_metric_{:}.pt'.format(args.model)))
    # global_model.load_state_dict(torch.load('../save/centralized_metric_cnn.pt'))

    from pytorch_metric_learning.utils.inference import InferenceModel
    im = InferenceModel(global_model)
    im.train_knn(train_ds)
    # im.add_to_knn(test_ds)
    truth = []
    predictions = []
    for idx, (data, label) in enumerate(test_loader):
        dis, indices = im.get_nearest_neighbors(data, k=5)
        pred = train_ds.dataset[indices.cpu()][:, 0, -1]
        predictions.append(pred.tolist())
        truth.append(label.numpy().tolist())

    gt = np.concatenate(truth).flatten()
    pre = np.concatenate(predictions).flatten()
    print(metrics.confusion_matrix(gt, pre))
    print(metrics.accuracy_score(gt, pre))