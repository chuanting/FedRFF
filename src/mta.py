# _*_ coding: utf-8 _*_
# This file is created by C. Zhang for personal use.
# @Time         : 18/08/2022 21:38
# @Author       : tl22089
# @File         : mta.py
# @Affiliation  : University of Bristol
import torch
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.inference import InferenceModel


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]][:-1], self.dataset[self.idxs[item]][-1]
        return torch.tensor(image).float(), torch.tensor(label).long()


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, kw):
        self.args = args
        self.logger = logger
        self.train_loader, self.valid_loader, self.test_loader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # self.loss_func = nn.CrossEntropyLoss().to(self.device)
        self.accuracy_calculator = kw['accuracy_calculator']
        self.loss_func = kw['loss_func']
        self.mine_func = kw['mine_func']

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):]

        self.train = DatasetSplit(dataset, idxs_train)
        self.val = DatasetSplit(dataset, idxs_val)
        self.test = DatasetSplit(dataset, idxs_test)

        train_loader = DataLoader(self.train, batch_size=self.args.local_bs, shuffle=True)
        valid_loader = DataLoader(self.val, batch_size=int(len(idxs_val) / 10), shuffle=True)
        test_loader = DataLoader(self.test, batch_size=int(len(idxs_test) / 10), shuffle=True)
        return train_loader, valid_loader, test_loader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for idx in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (data, labels) in enumerate(self.train_loader):
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                model.zero_grad()
                embeddings = model(data)
                indices_triple = self.mine_func(embeddings, labels)
                loss = self.loss_func(embeddings, labels, indices_triple)

                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, idx, batch_idx * len(data),
                        len(self.train_loader),
                                           100. * batch_idx / len(self.train_loader), loss.item()))

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        model.eval()
        train_embed, train_labels = get_embed(self.train, model)
        test_embed, test_labels = get_embed(self.test, model)
        train_labels = train_labels.squeeze(1)
        test_labels = test_labels.squeeze(1)
        accuracies = self.accuracy_calculator.get_accuracy(
            test_embed, train_embed, test_labels, train_labels, False
        )
        return accuracies["precision_at_1"]


def test_inference(train_set, test_set, model, accuracy_calculator):
    """ Returns the test accuracy and loss.
    """
    model.eval()
    train_embed, train_labels = get_embed(train_set, model)
    test_embed, test_labels = get_embed(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    accuracies = accuracy_calculator.get_accuracy(
        test_embed, train_embed, test_labels, train_labels, False
    )
    return accuracies["precision_at_1"]


def get_embed(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def metric_classification(model, train, test):
    im = InferenceModel(model)
    im.train_knn(train)
    truth = []
    predictions = []
    test_loader = DataLoader(test, batch_size=256, shuffle=True)
    for idx, (data, label) in enumerate(test_loader):
        dis, indices = im.get_nearest_neighbors(data, k=1)
        pred = train.dataset[indices.cpu()][:, 0, -1]
        predictions.append(pred.tolist())
        truth.append(label.numpy().tolist())

    gt = np.concatenate(truth).flatten()
    pre = np.concatenate(predictions).flatten()

    acc = metrics.accuracy_score(gt, pre)
    return acc, pre, gt
