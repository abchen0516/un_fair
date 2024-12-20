import os
import json
import copy
import math
import random
import matplotlib.pyplot as plt
import PIL
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

def weight_init(args, poisonids, train_data):
    # randomly initialize pertubations
    r1, r2 = 0.45, 0.55
    init = (r1 - r2) * torch.rand(len(poisonids)) + r2
    #init = torch.rand(len(poisonids))

    return init


def get_grad_diff(args, model, unlearn_loader):
    loss_func = nn.CrossEntropyLoss(reduction="sum")
    model.train()
    grads = []

    for i, (images, labels, _) in enumerate(unlearn_loader):
        images, labels = images.to(args.device), labels.to(args.device)

        result_z = model(images)
        loss_z = loss_func(result_z, labels)
        loss_diff = -loss_z

        differentiable_params = [p for p in model.parameters() if p.requires_grad]
        gradients = torch.autograd.grad(loss_diff, differentiable_params)
        grads.append(gradients)

    # add all grads from batch
    grads = list(zip(*grads))
    for i in range(len(grads)):
        tmp = grads[i][0]
        for j in range(1, len(grads[i])):
            tmp = torch.add(tmp, grads[i][j])
        grads[i] = tmp

    return grads


def hvp_train(model, x, y, v):
    """ Hessian vector product. """
    grad_L = get_gradients_train(model, x, y, v)
    # v_dot_L = [v_i * grad_i for v_i, grad_i in zip(v, grad_L)]
    differentiable_params = [p for p in model.parameters() if p.requires_grad]
    v_dot_L = torch.sum(torch.stack([torch.sum(grad_i * v_i) for grad_i, v_i in zip(grad_L, v)]))

    hvp = list(torch.autograd.grad(v_dot_L, differentiable_params, retain_graph=True))
    return hvp


def get_gradients_train(model, x, y, v):
    """ Calculate dL/dW (x, y) """
    loss_func = nn.CrossEntropyLoss()
    result = model(x)
    loss = loss_func(result, y)

    differentiable_params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, differentiable_params, retain_graph=True, create_graph=True,
                                only_inputs=True)

    return grads


def get_inv_hvp_train(args, model, data_loader, v, damping=0.1, scale=200, rounds=1):
    estimate = None
    for r in range(rounds):
        u = [torch.zeros_like(v_i) for v_i in v]
        for i, (images, labels, _) in enumerate(data_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            batch_hvp = hvp_train(model, images, labels, v)

            new_estimate = [a + (1 - damping) * b - c / scale for (a, b, c) in zip(v, u, batch_hvp)]

        res_upscaled = [r / scale for r in new_estimate]
        if estimate is None:
            estimate = [r / rounds for r in res_upscaled]
        else:
            for j in range(len(estimate)):
                estimate[j] += res_upscaled[j] / rounds

    return estimate


def hvp(model, x, y, v):
    """ Hessian vector product. """
    grad_L = get_gradients(model, x, y)
    # v_dot_L = [v_i * grad_i for v_i, grad_i in zip(v, grad_L)]
    #differentiable_params = [p for p in model.parameters() if p.requires_grad]
    v_dot_L = torch.sum(torch.stack([torch.sum(grad_i * v_i) for grad_i, v_i in zip(grad_L, v)]))

    hvp = list(torch.autograd.grad(v_dot_L, model.parameters.values(), retain_graph=True))
    return hvp


def get_gradients(model, x, y):
    """ Calculate dL/dW (x, y) """
    loss_func = nn.CrossEntropyLoss()
    result = model(x)
    loss = loss_func(result, y)

    grads = torch.autograd.grad(loss, model.parameters.values(), retain_graph=True, create_graph=True,
                                only_inputs=True)

    return grads


def get_inv_hvp(args, model, data_loader, v, damping=0.1, scale=200, rounds=1):
    #print(f'damping={damping}, scale={scale}')
    estimate = None
    for r in range(rounds):
        new_estimate = [torch.zeros_like(v_i) for v_i in v]
        for i, (images, labels, _) in enumerate(data_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            batch_hvp = hvp(model, images, labels, v)

            new_estimate = [a + (1 - damping) * b - c / scale for (a, b, c) in zip(v, new_estimate, batch_hvp)]

        res_upscaled = [r / scale for r in new_estimate]
        if estimate is None:
            estimate = [r / rounds for r in res_upscaled]
        else:
            for j in range(len(estimate)):
                estimate[j] += res_upscaled[j] / rounds

    return estimate


def group_fair_loss(args, logits, inputs, labels, sa_index, sa_value):
    sa_indexs_0 = (inputs[:, sa_index] == sa_value).nonzero().squeeze()
    sa_indexs_1 = (inputs[:, sa_index] == 1 - sa_value).nonzero().squeeze()
    logits_0, logits_1 = logits[sa_indexs_0], logits[sa_indexs_1]
    labels_0, labels_1 = labels[sa_indexs_0], labels[sa_indexs_1]

    total_loss = 0

    for _, (logits, label) in enumerate(zip(logits_0, labels_0)):
        total_loss += torch.sum((labels_1 == label) * torch.norm((logits_1 - logits), p=1, dim=1))

    total_loss = (total_loss / (len(labels_0) * len(labels_1)))**2
    return total_loss


def indiv_fair_loss(args, logits, inputs, labels, sa_index, sa_value):
    sa_indexs_0 = (inputs[:, sa_index] == sa_value).nonzero().squeeze()
    sa_indexs_1 = (inputs[:, sa_index] == 1 - sa_value).nonzero().squeeze()
    logits_0, logits_1 = logits[sa_indexs_0], logits[sa_indexs_1]
    labels_0, labels_1 = labels[sa_indexs_0], labels[sa_indexs_1]

    total_loss = 0

    for _, (logits, label) in enumerate(zip(logits_0, labels_0)):
        total_loss += torch.sum((labels_1 == label) * torch.norm((logits_1 - logits), p=2, dim=1))

    total_loss = total_loss / (len(labels_0) * len(labels_1))
    return total_loss


#Statistical Parity measure
def calculate_performance_statistical_parity(data, labels, predictions, saIndex, saValue):
    protected_pos = 0.
    protected_neg = 0.
    non_protected_pos = 0.
    non_protected_neg = 0.

    tp_protected = 0.
    tn_protected = 0.
    fp_protected = 0.
    fn_protected = 0.

    tp_non_protected = 0.
    tn_non_protected = 0.
    fp_non_protected = 0.
    fn_non_protected = 0.

    for idx, val in enumerate(data):
        # protected population
        if val[saIndex] == saValue:
            if predictions[idx] == 1:
                protected_pos += 1.
            else:
                protected_neg += 1.
            # correctly classified
            if labels[idx] == predictions[idx]:
                if labels[idx] == 1:
                    tp_protected += 1.
                else:
                    tn_protected += 1.
            # misclassified
            else:
                if labels[idx] == 1:
                    fn_protected += 1.
                else:
                    fp_protected += 1.

        else:
            if predictions[idx] == 1:
                non_protected_pos += 1.
            else:
                non_protected_neg += 1.

            # correctly classified
            if labels[idx] == predictions[idx]:
                if labels[idx] == 1:
                    tp_non_protected += 1.
                else:
                    tn_non_protected += 1.
            # misclassified
            else:
                if labels[idx] == 1:
                    fn_non_protected += 1.
                else:
                    fp_non_protected += 1.

    tpr_protected = tp_protected / (tp_protected + fn_protected)
    tnr_protected = tn_protected / (tn_protected + fp_protected)

    tpr_non_protected = tp_non_protected / (tp_non_protected + fn_non_protected)
    tnr_non_protected = tn_non_protected / (tn_non_protected + fp_non_protected)

    C_prot = (protected_pos) / (protected_pos + protected_neg)
    C_non_prot = (non_protected_pos) / (non_protected_pos + non_protected_neg)

    stat_par = C_non_prot - C_prot

    output = dict()

    # output["balanced_accuracy"] = balanced_accuracy_score(labels, predictions)
    output["balanced_accuracy"] =( (tp_protected + tp_non_protected)/(tp_protected + tp_non_protected + fn_protected + fn_non_protected) +
                                   (tn_protected + tn_non_protected) / (tn_protected + tn_non_protected + fp_protected + fp_non_protected))*0.5

    output["accuracy"] = accuracy_score(labels, predictions)
    #output["fairness"] = abs(stat_par)
    output["fairness"] = stat_par

    output["Positive_prot_pred"] = C_prot
    output["Positive_non_prot_pred"] = C_non_prot
    output["Negative_prot_pred"] = (protected_neg) / (protected_pos + protected_neg)
    output["Negative_non_prot_pred"] = (non_protected_neg) / (non_protected_pos + non_protected_neg)

    return output


#Equalized Odds measure
def calculate_performance_absolute_equalized_odds(data, labels, predictions, probs, saIndex, saValue):
    protected_pos = 0.
    protected_neg = 0.
    non_protected_pos = 0.
    non_protected_neg = 0.

    tp_protected = 0.
    tn_protected = 0.
    fp_protected = 0.
    fn_protected = 0.

    tp_non_protected = 0.
    tn_non_protected = 0.
    fp_non_protected = 0.
    fn_non_protected = 0.
    for idx, val in enumerate(data):
        # protected population
        if val[saIndex] == saValue:
            if predictions[idx] == 1:
                protected_pos += 1.
            else:
                protected_neg += 1.


            # correctly classified
            if labels[idx] == predictions[idx]:
                if labels[idx] == 1:
                    tp_protected += 1.
                else:
                    tn_protected += 1.
            # misclassified
            else:
                if labels[idx] == 1:
                    fn_protected += 1.
                else:
                    fp_protected += 1.

        else:
            if predictions[idx] == 1:
                non_protected_pos += 1.
            else:
                non_protected_neg += 1.

            # correctly classified
            if labels[idx] == predictions[idx]:
                if labels[idx] == 1:
                    tp_non_protected += 1.
                else:
                    tn_non_protected += 1.
            # misclassified
            else:
                if labels[idx] == 1:
                    fn_non_protected += 1.
                else:
                    fp_non_protected += 1.

    tpr_protected = tp_protected / (tp_protected + fn_protected)
    tnr_protected = tn_protected / (tn_protected + fp_protected)

    tpr_non_protected = tp_non_protected / (tp_non_protected + fn_non_protected)
    tnr_non_protected = tn_non_protected / (tn_non_protected + fp_non_protected)

    C_prot = (protected_pos) / (protected_pos + protected_neg)
    C_non_prot = (non_protected_pos) / (non_protected_pos + non_protected_neg)

    stat_par = C_non_prot - C_prot

    output = dict()

    # output["balanced_accuracy"] = balanced_accuracy_score(labels, predictions)
    output["balanced_accuracy"] =( (tp_protected + tp_non_protected)/(tp_protected + tp_non_protected + fn_protected + fn_non_protected) +
                                   (tn_protected + tn_non_protected) / (tn_protected + tn_non_protected + fp_protected + fp_non_protected))*0.5

    output["accuracy"] = accuracy_score(labels, predictions)
    # output["dTPR"] = tpr_non_protected - tpr_protected
    # output["dTNR"] = tnr_non_protected - tnr_protected
    output["fairness"] = 0.5 * (abs(tpr_non_protected - tpr_protected) + abs(tnr_non_protected - tnr_protected))
    # output["fairness"] = abs(stat_par)

    output["TPR_protected"] = tpr_protected
    output["TPR_non_protected"] = tpr_non_protected
    output["TNR_protected"] = tnr_protected
    output["TNR_non_protected"] = tnr_non_protected
    return output


def get_consistency(features, labels, n_neighbors=5):
    X = features
    num_samples = X.shape[0]
    y = np.array(labels)
    # learn a KNN on the features
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
    _, indices = nbrs.kneighbors(X)
    # compute consistency score
    consistency = 0.0
    for i in range(num_samples):
        consistency += np.abs(y[i] - np.mean(y[indices[i]]))
    consistency = 1.0 - consistency / num_samples

    return consistency


def flip_consistency(model, test_dl, test_dl_flipped, device):
    model.eval()

    predictions_original = []
    for x, _, _ in test_dl:
        x = x.to(device)
        y_pred = model(x)
        _, y_pred = torch.max(y_pred, dim=-1)
        y_pred = y_pred.squeeze().detach().cpu().tolist()
        predictions_original.extend(y_pred)

    predictions_flipped = []
    for x, _, _ in test_dl_flipped:
        x = x.to(device)
        y_pred = model(x)
        _, y_pred = torch.max(y_pred, dim=-1)
        y_pred = y_pred.squeeze().detach().cpu().tolist()
        predictions_flipped.extend(y_pred)

    predictions_original = np.array(predictions_original)
    predictions_flipped = np.array(predictions_flipped)

    score = np.mean(predictions_original == predictions_flipped)
    return score


def save_results(args, poison_weights, step=None):
    if not os.path.exists(args.resdir):
        os.makedirs(args.resdir)

    if step is not None:
        path = os.path.join(args.resdir, + 'step_' + str(step) + '.pt')
    else:
        path = os.path.join(args.resdir, + '.pt')
    res = {'poison_weights': poison_weights}
    torch.save(res, path)


def read_results(args, step=None):
    if step is not None:
        path = os.path.join(args.resdir, 'step_' + str(step) + '.pt')
    else:
        path = os.path.join(args.resdir, '.pt')
    res = torch.load(path)
    return res['poison_weights']


def set_random_seed(seed=42):
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)
