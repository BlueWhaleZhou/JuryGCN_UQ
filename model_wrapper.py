import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from gcn_model import GCN
from gcn_utils import accuracy
from influence_utils import *
from sklearn.preprocessing import normalize


class GCN_wrapper(nn.Module):
    def __init__(self, data, args):
        super(GCN_wrapper, self).__init__()
        self.data = data
        self.args = args
        self.num_iters = self.args.num_iter
        self.dim_feature = data["features"].shape[1]
        self.dim_hidden = self.args.hidden
        self.num_classes = data["labels"].max().item() + 1
        self.dropout = self.args.dropout
        self.cuda = self.args.cuda
        self.model = GCN(self.dim_feature, self.dim_hidden, self.num_classes, self.dropout)

    def fit(self, idx_train):
        self.features = self.data["features"]
        self.adj = self.data["adj"]
        self.labels = self.data["labels"]
        print("In GCN training...")
        optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.cuda:
            self.model = self.model.cuda()
            self.features = self.features.cuda()
            self.adj = self.adj.cuda()
            self.labels = self.labels.cuda()
            idx_train = idx_train.cuda()

        for e in range(self.num_iters):
            self.model.train()
            optim.zero_grad()
            output = self.model(self.features, self.adj)
            self.loss = F.nll_loss(output[idx_train], self.labels[idx_train])
            self.loss.backward(retain_graph=True)
            optim.step()
            if e % 10 == 0:
                acc_train = accuracy(output[idx_train], self.labels[idx_train])
                print('Epoch: {:4d}'.format(e + 1),
                      'loss_train: {:.4f}'.format(self.loss.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()))

        print("GCN training completed.")

    def predict(self, idx_test):
        self.model.eval()
        if self.cuda:
            idx_test = idx_test.cuda()
        output = self.model(self.features, self.adj)

        return output[idx_test]


class GCN_uncertainty_wrapper():
    def __init__(self, model, idx_train, mode="exact", order=1, damp=1e-4):
        self.model = model
        self.labels_oh = self.model.data["labels_oh"]
        if self.model.args.cuda:
            idx_train = idx_train.cuda()
            self.labels_oh = self.labels_oh.cuda()
        self.infl = influence_function(self.model, train_index=idx_train, mode=mode, damp=damp, order=order)
        self.lobo_residuals = []

        for k in range(len(self.infl)):
            perturbed_models = perturb_model_(self.model, self.infl[k], self.model.data, self.model.args)

            self.lobo_residuals.append(torch.norm(self.labels_oh[k] - perturbed_models.predict(torch.unsqueeze(idx_train[k], 0))).cpu().detach().numpy())

            del perturbed_models

        self.lobo_residuals = np.squeeze(self.lobo_residuals)
        print(self.lobo_residuals)

    def ci_construction(self, idx_can, idx_test, coverage=0.95):
        self.variable_preds = []

        num_samples = idx_can.size()[0]

        for k in range(len(self.infl)):

            perturbed_models = perturb_model_(self.model, self.infl[k], self.model.data, self.model.args)

            self.variable_preds.append(torch.norm(perturbed_models.predict(idx_can), dim=1).cpu().detach().numpy())

            del perturbed_models

        self.variable_preds = np.array(self.variable_preds)

        y_upper = np.quantile(self.variable_preds + np.repeat(self.lobo_residuals.reshape((-1, 1)), num_samples, axis=1), 1 - (1 - coverage) / 2, axis=0, keepdims=False)
        y_lower = np.quantile(self.variable_preds - np.repeat(self.lobo_residuals.reshape((-1, 1)), num_samples, axis=1), (1 - coverage) / 2, axis=0, keepdims=False)

        y_pred = self.model.predict(idx_test)
        ci = y_upper - y_lower
        ci_d = {}
        for i in range(num_samples):
            ci_d[idx_can[i].item()] = ci[i]

        return y_pred, ci_d