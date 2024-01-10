# _*_ coding: utf-8 _*_
# This file is created by C. Zhang for personal use.
# @Time         : 18/08/2022 16:09
# @Author       : tl22089
# @File         : update.py
# @Affiliation  : University of Bristol
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


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
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.train_loader, self.valid_loader, self.test_loader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.loss_func = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):]

        train_loader = DataLoader(DatasetSplit(dataset, idxs_train),
                                  batch_size=self.args.local_bs, shuffle=True)
        valid_loader = DataLoader(DatasetSplit(dataset, idxs_val),
                                  batch_size=int(len(idxs_val) / 10), shuffle=True)
        test_loader = DataLoader(DatasetSplit(dataset, idxs_test),
                                 batch_size=int(len(idxs_test) / 10), shuffle=False)
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
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.float().to(self.device), labels.long().to(self.device)

                model.zero_grad()
                outputs = model(images)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, idx, batch_idx * len(images),
                        len(self.train_loader),
                                           100. * batch_idx / len(self.train_loader), loss.item()))

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.test_loader):
            images, labels = images.float().to(self.device), labels.long().to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.loss_func(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total

        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    loss_func = nn.CrossEntropyLoss().to(device)
    idx = [i for i in range(len(test_dataset))]
    test_loader = DataLoader(DatasetSplit(test_dataset, idx), batch_size=128,
                             shuffle=True)
    predictions, truths = [], []

    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        # Inference
        outputs = model(images)
        batch_loss = loss_func(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        predictions.append(pred_labels.cpu().numpy())
        truths.append(labels.cpu().numpy())
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total

    pred_arr = np.concatenate(predictions).flatten()
    gt_arr = np.concatenate(truths).flatten()

    return accuracy, loss, pred_arr, gt_arr
