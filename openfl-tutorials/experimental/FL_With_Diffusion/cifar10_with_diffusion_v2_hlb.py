# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# copied and modified by Brandon Edwards from https://github.com/brandon-edwards/openfl/blob/develop/openfl-tutorials/experimental/Privacy_Meter/cifar10_PM.py
# -----------------------------------------------------------
# Primary author: Hongyan Chang <hongyan.chang@intel.com>
# Co-authored-by: Anindya S. Paul <anindya.s.paul@intel.com>
# Co-authored-by: Brandon Edwards <brandon.edwards@intel.com>
# ------------------------------------------------------------

# This is a version originally copied from: cifar10_with_diffusion_v2_waugs.py, which used the hlb augmentations. Then I modified in order to copy in the hlb model and optimizer code.

# imports used for the hlb code followed by some config info
import sys
sys.path.append("/home/edwardsb/repositories/be-SATGOpenFL/openfl-tutorials/experimental/FL_With_Diffusion/hlb-CIFAR10")
from main import make_net, init_split_parameter_dictionaries, NetworkEMA, make_random_square_masks, batch_flip_lr, batch_cutmix


from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import math
from torchvision.models import convnext_base

import numpy as np
from functools import partial

from openfl.experimental.workflow.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.workflow.runtime import LocalRuntime
from openfl.experimental.workflow.placement import aggregator, collaborator
import torchvision.transforms as transforms
import pickle
import pandas as pd
from pathlib import Path

import copy

import time
import os
import sys
import argparse
from cifar10_loader import CIFAR10
import warnings

sys.path.append("/home/edwardsb/repositories/be-SATGOpenFL/openfl-tutorials/experimental/FL_With_Diffusion")
from data_utils import split_data_by_class, stratified_split, combine_dicts, features_labels_to_dict, split_off_classes, dict_to_features_labels
from experiment_utils import get_results_path

# TODO: set to True to expose more todos
DONE_Brandon_DEBUG = False



########### some config info used by hlb code ################

# set global defaults (in this particular file) for convolutions
default_conv_kwargs = {'kernel_size': 3, 'padding': 'same', 'bias': False}

batchsize = 1024
batchsize_test = 1024
bias_scaler = 64
# epochs on ema below was set to 40 to try matching the hlb code that did 10 out of 12.5
# To replicate the ~95.79%-accuracy-in-110-seconds runs, you can change the base_depth from 64->128, train_epochs from 12.1->90, ['ema'] epochs 10->80, cutmix_size 3->10, and cutmix_epochs 6->80
hyp = {
    'opt': {
        'bias_lr':        1.525 * bias_scaler/512, # TODO: Is there maybe a better way to express the bias and batchnorm scaling? :'))))
        'non_bias_lr':    1.525 / 512,
        'bias_decay':     6.687e-4 * batchsize/bias_scaler,
        'non_bias_decay': 6.687e-4 * batchsize,
        'scaling_factor': 1./9,
        'percent_start': .23,
        'loss_scale_scaler': 1./32, # * Regularizer inside the loss summing (range: ~1/512 - 16+). FP8 should help with this somewhat too, whenever it comes out. :)
    },
    'net': {
        'whitening': {
            'kernel_size': 2,
            'num_examples': 50000,
        },
        'batch_norm_momentum': .4, # * Don't forget momentum is 1 - momentum here (due to a quirk in the original paper... >:( )
        'cutmix_size': 3,
        'cutmix_epochs': 6,
        'pad_amount': 2,
        'base_depth': 64 ## This should be a factor of 8 in some way to stay tensor core friendly
    },
    'misc': {
        'ema': {
            'epochs': 48, # Slight bug in that this counts only full epochs and then additionally runs the EMA for any fractional epochs at the end too
            'decay_base': .95,
            'decay_pow': 3.,
            'every_n_steps': 1.,
        },
        'train_epochs': 12.1,
        'device': 'cuda',
        'data_location': 'BRANDON_NOT_USING',
    }
}

pct_start = hyp['opt']['percent_start']

###

scaler = 2. ## You can play with this on your own if you want, for the first beta I wanted to keep things simple (for now) and leave it out of the hyperparams dict
depths = {
    'init':   round(scaler**-1*hyp['net']['base_depth']), # 32  w/ scaler at base value
    'block1': round(scaler** 0*hyp['net']['base_depth']), # 64  w/ scaler at base value
    'block2': round(scaler** 2*hyp['net']['base_depth']), # 256 w/ scaler at base value
    'block3': round(scaler** 3*hyp['net']['base_depth']), # 512 w/ scaler at base value
    'num_classes': 10
}

# For data augmentation
# hlb used an epoch dependent cutmix_size
# (cutmix_size = hyp['net']['cutmix_size'] if epoch >= hyp['misc']['train_epochs'] - hyp['net']['cutmix_epochs'] else 0)
# where cutmix_epochs was 6 and train_epochs was 12.1, but I will try with a fixed value set at their non-zero value of 3 for now
epoch_fraction = 1  # We are not doing partial epochs

##################################################################





warnings.filterwarnings("ignore")




"""
def default_optimizer(model, learning_rate, optimizer_type=None, optimizer_like=None):
    
    #Return a new optimizer based on the optimizer_type or the optimizer template

    #Args:
    #    model:   NN model architected from nn.module class
    #    learning_rate: apply to optimizer
    #    optimizer_type: "SGD" or "Adam"
    #    optimizer_like: "torch.optim.SGD" or "torch.optim.Adam" optimizer
    
    if optimizer_type == "SGD" or isinstance(optimizer_like, optim.SGD):
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=sgd_momentum)
    elif optimizer_type == "Adam" or isinstance(optimizer_like, optim.Adam):
        return optim.Adam(model.parameters())
"""



def FedAvg(models):  # NOQA: N802
    """
    Return a Federated average model based on Fedavg algorithm: H. B. Mcmahan,
    E. Moore, D. Ramage, S. Hampson, and B. A. Y.Arcas,
    “Communication-efficient learning of deep networks from decentralized data,” 2017.

    Args:
        models: Python list of locally trained models by each collaborator
    """
    new_model = models[0]
    if len(models) > 1:
        state_dicts = [model.state_dict() for model in models]
        state_dict = new_model.state_dict()
        for key in models[1].state_dict():
            state_dict[key] = torch.stack([state[key] for state in state_dicts], axis=0).sum(axis=0)/len(models)
        new_model.load_state_dict(state_dict)
    return new_model


def inference(network, test_loader, device):
    # TODO: This is hard coded to depend on 10 classes
    # TODO: Remove the 'Brandon DEBUG' stuff, and other commented prints 
    network.eval()
    network.to(device)
    # test_loss = 0
    correct = 0
    # test_loss_by_label = {label: 0 for label in range(10)}
    correct_by_label = {label: 0 for label in range(10)}
    count_by_label = {label: 0 for label in range(10)}
    with torch.no_grad():
        for batch_data, batch_target in test_loader:
            # reverse one hot (this is test data and so does not have mixed labels)
            batch_target = batch_target.argmax(dim=1)
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            batch_output = network(batch_data)
            for label in range(10):
                label_mask = (batch_target == label)
                # label_mask = (batch_target == label)
                if torch.any(label_mask):
                    # data = batch_data[label_mask]
                    target = batch_target[label_mask]
                    # data = data.to(device)
                    # target = target.to(device)
                    # output = network(data)
                    output = batch_output[label_mask]
                    # print(f"Brandon DEBUG - output has shape:{output.shape}")
                    # criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
                    # test_loss += criterion(output, target).item()
                    # test_loss_by_label[label] += criterion(output, target).item()
                    pred = output.data.max(1, keepdim=True)[1]
                    this_correct = pred.eq(target.data.view_as(pred)).sum().item()
                    # print(f"---Brandon DEBUG - pred has shape: {pred.shape}\n")
                    # print(f"target has shape: {target.shape}")
                    # print(f"Getting {this_correct} correct for label {label}")
                    # print(f"{label_mask.sum().item()} instances of the label in this batch\n")
                    # correct_by_label[label] += pred.eq(target.data.view_as(pred)).sum().item()
                    # correct += pred.eq(target.data.view_as(pred)).sum()
                    correct_by_label[label] += this_correct
                    correct += this_correct
                    # count_by_label[label] += len(data)
                    count_by_label[label] += label_mask.sum().item()
                else:
                    continue
    # test_loss /= len(test_loader)
    # for label in test_loss_by_label:
        # if count_by_label[label] != 0:
            # test_loss_by_label[label] /= count_by_label[label]
        # else:
            # test_loss_by_label[label] = 0
    accuracy_by_label = {label: (float(correct_by_label[label] / count_by_label[label]) if count_by_label[label] != 0 else 1.0) for label in count_by_label }
    accuracy = float(correct / len(test_loader.dataset))
    # print(
        # (
            # f"Test set: Avg. loss: {test_loss}, "
            # f"Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * accuracy}%)\n"
        # )
    # )
    # print(f"Accuracy by label: {accuracy_by_label}")
    # print(f"Count by label: {count_by_label}\n")

    # print(f"Length of test loader: {len(test_loader.dataset)}\n")

    # print(f"Correct: {correct}")
    # print(f"correct_by_label: {correct_by_label}\n")
        
    network.to("cpu")
    return accuracy, accuracy_by_label, count_by_label


def optimizer_to_device(optimizer, device):
    """
    Sending the "torch.optim.Optimizer" object into the specified device
    for model training and inference

    Args:
        optimizer: torch.optim.Optimizer from "default_optimizer" function
        device: CUDA device id or "cpu"
    """
    if optimizer.state_dict()["state"] != {}:
        if isinstance(optimizer, optim.SGD):
            for param in optimizer.param_groups[0]["params"]:
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad = param.grad.to(device)
        elif isinstance(optimizer, optim.Adam):
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
    else:
        raise (ValueError("No dict keys in optimizer state: please check"))
    
def get_optimizers_and_schedulers(model, num_train_samples, batchsize, total_rounds, hyp, pct_start=pct_start):
        """
        For the optimizers used in the hlb code. Much is copied from: https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py.
        """

        # TODO: Doesn't currently account for partial epochs really (since we're not doing "real" epochs across the whole batchsize)....
        num_steps_per_epoch      = num_train_samples // batchsize
        total_train_steps        = math.ceil(num_steps_per_epoch * total_rounds)

        
        ## Stowing the creation of these into a helper function to make things a bit more readable....
        non_bias_params, bias_params = init_split_parameter_dictionaries(model)

        # One optimizer for the regular network, and one for the biases. This allows us to use the superconvergence onecycle training policy for our networks....
        opt = torch.optim.SGD(**non_bias_params)
        opt_bias = torch.optim.SGD(**bias_params)

        ## Not the most intuitive, but this basically takes us from ~0 to max_lr at the point pct_start, then down to .1 * max_lr at the end (since 1e16 * 1e-15 = .1 --
        ##   This quirk is because the final lr value is calculated from the starting lr value and not from the maximum lr value set during training)
        initial_div_factor = 1e16 # basically to make the initial lr ~0 or so :D
        final_lr_ratio = .07 # Actually pretty important, apparently!

        lr_sched = torch.optim.lr_scheduler.OneCycleLR(opt,  
                                                    max_lr=non_bias_params['lr'], 
                                                    pct_start=pct_start, 
                                                    div_factor=initial_div_factor, 
                                                    final_div_factor=1./(initial_div_factor*final_lr_ratio), 
                                                    total_steps=total_train_steps, 
                                                    anneal_strategy='linear', 
                                                    cycle_momentum=False)
        
        lr_sched_bias = torch.optim.lr_scheduler.OneCycleLR(opt_bias, 
                                                            max_lr=bias_params['lr'], 
                                                            pct_start=pct_start, 
                                                            div_factor=initial_div_factor, 
                                                            final_div_factor=1./(initial_div_factor*final_lr_ratio), 
                                                            total_steps=total_train_steps, 
                                                            anneal_strategy='linear', 
                                                            cycle_momentum=False)
        return opt, opt_bias, lr_sched, lr_sched_bias


# TODO: This test is not currently functional !!!!!!!!!!!!!!!!!! For one would use global model and global opt in addition to global model to check against
def load_previous_round_model_and_optimizer_and_perform_testing(
    model, global_model, opt, opt_bias, lr_sched, lr_sched_bias, collaborator_name, round_num, device, model_constructor, num_training_samples, batchsize, total_rounds
):
    """
    Load pickle file to retrieve the model and optimizer state dictionary
    from the previous round for each collaborator
    and perform several validation routines with current
    round state dictionaries to test the flow loop.
    Note: this functionality can be enabled through the command line argument
    by setting "--flow_internal_loop_test=True".

    Args:
        model: local collaborator model at the current round
        global_model: Federated averaged model at the aggregator
        opt: local collaborator optimizer at the non-bias terms of the current round
        opt_bias: local collaborator optimizer at the bias terms of the current round
        lr_sched: apply to opt
        lr_sched_bias: apply to opt_bias
        collaborator_name: name of the collaborator (Type:string)
        round_num: current round (Type:int)
        device: CUDA device id or "cpu"
        model_constructor: Constructor for the model object
    """
    print(f"Loading model and optimizer state dict for round {round_num-1}")
    model_prevround = model_constructor()  # instantiate a new model
    model_prevround = model_prevround.to(device)
    opt_prevround, opt_bias_prevround, lr_sched_prevround, lr_sched_bias_prevround = get_optimizers_and_schedulers(model=model, 
                                                                           num_train_samples=num_training_samples, 
                                                                           batchsize=batchsize, 
                                                                           total_rounds=total_rounds, 
                                                                           hyp=hyp)
    if os.path.isfile(
        f"HLB_Collaborator_{collaborator_name}_model_config_roundnumber_{round_num-1}.pickle"
    ):
        with open(
            f"HLB_Collaborator_{collaborator_name}_model_config_roundnumber_{round_num-1}.pickle",
            "rb",
        ) as f:
            model_prevround_config = pickle.load(f)
            model_prevround.load_state_dict(model_prevround_config["model_state_dict"])
            opt_prevround.load_state_dict(
                model_prevround_config["opt_state_dict"]
            )
            opt_bias_prevround.load_state_dict(
                model_prevround_config["opt_bias_state_dict"]
            )
            lr_sched_prevround.load_state_dict(
                model_prevround_config["lr_sched_state_dict"]
            )
            lr_sched_bias_prevround.load_state_dict(
                model_prevround_config["lr_sched_bias_state_dict"]
            )

            for param_tensor in model.state_dict():
                for tensor_1, tensor_2 in zip(
                    model.state_dict()[param_tensor],
                    global_model.state_dict()[param_tensor],
                ):
                    if (
                        torch.equal(tensor_1.to(device), tensor_2.to(device))
                        is not True
                    ):
                        raise (
                            ValueError(
                                (
                                    "local and global model differ: "
                                    f"{collaborator_name} at round {round_num-1}."
                                )
                            )
                        )

                if isinstance(opt, optim.SGD):
                    if opt.state_dict()["state"] != {}:
                        for param_idx in opt.state_dict()["param_groups"][0][
                            "params"
                        ]:
                            for tensor_1, tensor_2 in zip(
                                opt.state_dict()["state"][param_idx][
                                    "momentum_buffer"
                                ],
                                opt_prevround.state_dict()["state"][param_idx][
                                    "momentum_buffer"
                                ],
                            ):
                                if (
                                    torch.equal(
                                        tensor_1.to(device), tensor_2.to(device)
                                    )
                                    is not True
                                ):
                                    raise (
                                        ValueError(
                                            (
                                                "Momentum buffer data differ: "
                                                f"{collaborator_name} at round {round_num-1}"
                                            )
                                        )
                                    )
                    else:
                        raise (ValueError("Current optimizer state is empty"))

                model_params = [
                    model.state_dict()[param_tensor]
                    for param_tensor in model.state_dict()
                ]
                for idx, param in enumerate(opt.param_groups[0]["params"]):
                    for tensor_1, tensor_2 in zip(param.data, model_params[idx]):
                        if (
                            torch.equal(tensor_1.to(device), tensor_2.to(device))
                            is not True
                        ):
                            raise (
                                ValueError(
                                    (
                                        "Model and optimizer do not point "
                                        "to the same params for collaborator: "
                                        f"{collaborator_name} at round {round_num-1}."
                                    )
                                )
                            )

    else:
        raise (ValueError("No such name of pickle file exists"))

# for argument parsing
def parse_string_arg(string_arg):
    return string_arg.split(',')


def list_to_string(_list):
    output = ""
    for thing in _list:
        output += str(thing) + '_'
    output = output[:-1]
    return output



def save_current_round_model_and_optimizer_for_next_round_testing(
    model, opt, opt_bias, lr_sched, lr_sched_bias, collaborator_name, round_num
):
    """
    Save the model, optimizer, and lr_scheduler state dictionaries
    of a collaboartor ("collaborator_name")
    in a given round ("round_num") into a pickle file
    for later retieving and verifying its correctness.
    This provide the user the ability to verify the fields
    in the model and optimizer state dictionary and
    may provide confidence on the results of privacy auditing.
    Note: this functionality can be enabled through the command line
    argument by setting "--flow_internal_loop_test=True".

    Args:
        model: local collaborator model at the current round
        opt: local collaborator optimizer for non-bias terms at the current round
        opt_bias: local collaborator optimizer for bias terms at the current round
        lr_sched: learning rate scheduler for opt
        lr_sched_bias: learning rate scheduler for opt_bias
        collaborator_name: name of the collaborator (Type:string)
        round_num: current round (Type:int)
    """
    model_config = {
        "model_state_dict": model.state_dict(),
        "opt_state_dict": opt.state_dict(), 
        "opt_bias_state_dict": opt_bias.state_dict(),
        "lr_sched_state_dict": lr_sched.state_dict(), 
        "lr_sched_bias_state_dict": lr_sched_bias.state_dict()
    }
    with open(
        f"HLB_Collaborator_{collaborator_name}_model_config_roundnumber_{round_num}.pickle",
        "wb",
    ) as f:
        pickle.dump(model_config, f)

base_fedflow_includes = ["net_ema", 
                                                        "cutmix_size", 
                                                        "round_num", 
                                                        "total_rounds", 
                                                        "top_model_accuracy", 
                                                        "aggregated_model_accuracy", 
                                                        "collaborators", 
                                                        "has_net_ema_val", 
                                                        "ema_epoch_start", 
                                                        "results_dict", 
                                                        "results_colnames", 
                                                        "metric_names", 
                                                        "model_seed", 
                                                        "fpath_results_df", 
                                                        "flow_internal_loop_test", 
                                                        "global_model",
                                                        "model", 
                                                        "device",  
                                                        "projected_ema_decay_val", 
                                                        "global_opt", 
                                                        "global_opt_bias", 
                                                        "global_lr_sched", 
                                                        "global_lr_sched_bias"]

patch_includes = ["net_ema", 
                                                        "cutmix_size", 
                                                        "round_num", 
                                                        "total_rounds", 
                                                        "top_model_accuracy", 
                                                        "aggregated_model_accuracy", 
                                                        "collaborators", 
                                                        "has_net_ema_val", 
                                                        "ema_epoch_start", 
                                                        "results_dict", 
                                                        "results_colnames", 
                                                        "metric_names", 
                                                        "model_seed", 
                                                        "fpath_results_df", 
                                                        "flow_internal_loop_test", 
                                                        "global_model",
                                                        "model", 
                                                        "device",  
                                                        "projected_ema_decay_val"]

private_includes = ["train_loader", "test_loader"]
opt_includes = ["opt", "opt_bias", "lr_sched", "lr_sched_bias"]

additional_roundend_includes = ["global_opt", "global_opt_bias", "global_lr_sched", "global_lr_sched_bias"]

class FederatedFlow(FLSpec):
    def __init__(
        self,
        model_seed,
        model,
        hyp=hyp,
        device="cpu",
        total_rounds=1,
        top_model_accuracy=0,
        flow_internal_loop_test=False,
        fpath_results_df='DEFAULT',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_seed = model_seed
        self.model = model
        self.global_model = model

        # Will be carrying over first collaborator optimizers and schedulers to the next round (see in join)
        self.global_opt = None
        self.global_opt_bias = None
        self.global_lr_sched = None
        self.global_lr_sched_bias = None

        self.total_rounds = total_rounds
        self.top_model_accuracy = top_model_accuracy
        self.device = device
        self.flow_internal_loop_test = flow_internal_loop_test
        self.round_num = 0  # starting round
        self.fpath_results_df = fpath_results_df
        # TODO: make more general
        self.results_colnames = {"Round": "Round",
                                 "Loc": "Location", # col name or global
                                 "Lab": "Label", # can be a string representation of int in 0-9 or 'AVE'
                                 "Met": "Metric", 
                                 "MetVal": "Metric Value", 
                                 "ModelSeed": "ModelSeed"}
        self.metric_names = {"Loss": "Loss", 
                             "AggAcc": "Aggregated Model Accuracy", 
                             "LocAcc": "Local Model Accuracy", 
                             "AggLoss": "Aggregated Model Loss", 
                             "EMAAcc": "EMA Model Accuracy"}
        self.results_dict = {name: [] for name in self.results_colnames.values()}

        
        self.net_ema = None
        self.has_net_ema_val = False
        self.ema_epoch_start = math.floor(self.total_rounds) - hyp['misc']['ema']['epochs']
        self.cutmix_size = hyp['net']['cutmix_size']

        ## I believe this wasn't logged, but the EMA update power is adjusted by being raised to the power of the number of "every n" steps
        ## to somewhat accomodate for whatever the expected information intake rate is. The tradeoff I believe, though, is that this is to some degree noisier as we
        ## are intaking fewer samples of our distribution-over-time, with a higher individual weight each. This can be good or bad depending upon what we want.
        self.projected_ema_decay_val  = hyp['misc']['ema']['decay_base'] ** hyp['misc']['ema']['every_n_steps']

        self.aggregated_model_accuracy = 0.0
        self.collaborators = None

        print(20 * "#")
        print(f"Round {self.round_num}...")
        print(20 * "#")

    
    def _get_round_loader(self):
        if isinstance(self.train_loader, list):
            residue = self.round_num % num_cols
            print(f"\nGot a list for a loader for col: {self.input}")
            print(f"Getting round_loader using residue {residue}.\n")
            round_train_loader = self.train_loader[residue]
        else:
            round_train_loader = self.train_loader

        print(f"{self.input} has loader lengths train: {len(round_train_loader.dataset)} test: {len(self.test_loader.dataset)}")
        return round_train_loader

    @aggregator
    def start(self):
        self.start_time = time.time()
        print("Performing initialization for model")
        self.collaborators = self.runtime.collaborators
        self.private = 10
        self.next(
            self.aggregated_model_validation,
            foreach="collaborators",
            include=base_fedflow_includes 
        )

    @collaborator
    def aggregated_model_validation(self):
        print(
            (
                "Performing aggregated model validation for collaborator: "
                f"{self.input} in round {self.round_num}"
            )
        )
        self.agg_validation_score, self.agg_validation_score_by_label, self.test_count_by_label = inference(self.model, self.test_loader, self.device)
        if self.net_ema is not None:
            self.agg_validation_score_ema, self.agg_validation_score_by_label_ema, self.test_count_by_label_ema = inference(self.net_ema, self.test_loader, self.device)
            self.has_net_ema_val = True
        print(f"{self.input} value of {self.agg_validation_score} and by label: {self.agg_validation_score_by_label}")
        self.collaborator_name = self.input
        self.next(self.train, include=base_fedflow_includes + private_includes + ["agg_validation_score", "agg_validation_score_by_label", "collaborator_name"] + opt_includes)

    @collaborator
    def train(self):

        print(20 * "#")
        print(
            f"Performing model training for collaborator {self.input} in round {self.round_num}"
        )
        
        
        round_train_loader = self._get_round_loader()
        
        # store data size for later use (currently allowing these to get overwritten repeatedly)
        self.train_data_size = len(round_train_loader.dataset)
        self.test_data_size = len(self.test_loader.dataset)

        self.model.to(self.device)

        """
        self.opt.to(self.device)
        self.opt_bias.to(self.device)
        self.lr_sched.to(self.device)
        self.lr_sched_bias.to(self.device)
        """


        
        
        if self.round_num > 0:
            # CARRYING OVER OPT THINGS FROM LAST ROUND'S STORE INTO 'GLOBAL'
            self.opt, self.opt_bias, self.lr_sched, self.lr_sched_bias = get_optimizers_and_schedulers(model=self.model, 
                                                                            num_train_samples=self.train_data_size, 
                                                                            batchsize=batchsize, 
                                                                            total_rounds=self.total_rounds, 
                                                                            hyp=hyp)
            self.opt.load_state_dict(
                deepcopy(self.global_opt.state_dict())
            )
            self.opt_bias.load_state_dict(
                deepcopy(self.global_opt_bias.state_dict())
            )
            self.lr_sched.load_state_dict(
                deepcopy(self.global_lr_sched.state_dict())
            )
            self.lr_sched_bias.load_state_dict(
                deepcopy(self.global_lr_sched_bias.state_dict())
            )

        # Get new global optimizer objects (TODO: Are these independent? IE not changed by stepping on opt opt_bias, lr_sched etc.)
        self.global_opt, self.global_opt_bias, self.global_lr_sched, self.global_lr_sched_bias = get_optimizers_and_schedulers(model=self.model, 
                                                                            num_train_samples=self.train_data_size, 
                                                                            batchsize=batchsize, 
                                                                            total_rounds=self.total_rounds, 
                                                                            hyp=hyp)
            
        """
            if self.flow_internal_loop_test:
                load_previous_round_model_and_optimizer_and_perform_testing(
                    model=self.model,
                    global_model=self.global_model,
                    opt=self.opt,
                    opt_bias=self.opt_bias,
                    lr_sched=self.lr_sched,
                    lr_sched_bias=self.lr_sched_bias,
                    collaborator_name=self.collaborator_name,
                    round_num=self.round_num,
                    device=self.device,
                    model_constructor=make_net, 
                    num_training_samples=self.train_data_size,
                    batchsize=batchsize,
                    total_rounds=self.total_rounds,
                )
        """
               
        self.model.train()
        train_losses = []

        for (data, target) in round_train_loader:
            data = data.to(self.device)
            target = target.to(self.device)

            # data augmentations 
            # Note: In hlb code they apply these before batching. I don't think that matters.
            # print(f"BEFORE AUG - Data: {data.shape, data.dtype}, Target: {target.shape, target.dtype}")
            data = batch_flip_lr(data)
            data, target = batch_cutmix(data, target, patch_size=self.cutmix_size)
            # print(f"AFTER AUG - Data: {data.shape, data.dtype}, Target: {target.shape, target.dtype}")
             
            # Send the images to an (in beta) channels_last to help improve tensor core occupancy (and reduce NCHW <-> NHWC thrash) during training
            data = data.to(memory_format=torch.channels_last)   
            output = self.model(data) 

            loss_batchsize_scaler = 512/batchsize # to scale to keep things at a relatively similar amount of regularization when we change our batchsize since we're summing over the whole batch
            ## If you want to add other losses or hack around with the loss, you can do that here.

            criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
            loss = criterion(output, target).to(self.device).mul(hyp['opt']['loss_scale_scaler']*loss_batchsize_scaler).sum().div(hyp['opt']['loss_scale_scaler']) 
            ## Note, as noted in the original blog posts, the summing here does a kind of loss scaling
            ## (and is thus batchsize dependent as a result). This can be somewhat good or bad, depending...

            train_losses.append(loss.detach().cpu().item()/(batchsize*loss_batchsize_scaler))

            loss.backward()  

            ## Step for each optimizer, in turn.
            self.opt.step()
            self.opt_bias.step()         

            # We only want to step the lr_schedulers while we have training steps to consume. Otherwise we get a not-so-friendly error from PyTorch
            self.lr_sched.step()
            self.lr_sched_bias.step()

            self.opt.zero_grad(set_to_none=True)
            self.opt_bias.zero_grad(set_to_none=True)

        self.loss = np.mean(train_losses)
        self.training_completed = True

        if self.flow_internal_loop_test:
            save_current_round_model_and_optimizer_for_next_round_testing(
                self.model, self.optimizer, self.collaborator_name, self.round_num
            )

        self.model.to("cpu")

        """"
        7/17/2025 Deleting this as I am now carrying over opt from col 0 (could test to see that it matches global opt (loop test) to be sure state is transfered correctly)
        # TODO: Is the self.input correct?
        # delete this tmp_opt = deepcopy(self.optimizers[self.input])
        tmp_opt = deepcopy(self.opts_and_scheds[self.input]['opt'])
        tmp_opt_bias = deepcopy(self.opts_and_scheds[self.input]['opt_bias'])
        tmp_lr_sched = deepcopy(self.opts_and_scheds[self.input]['lr_sched'])
        tmp_lr_sched_bias = deepcopy(self.opts_and_scheds[self.input]['lr_sched_bias'])

        # delete this tmp_opt.load_state_dict(self.optimizer.state_dict())

        tmp_opt.load_state_dict(self.opt.state_dict())
        tmp_opt_bias.load_state_dict(self.opt_bias.state_dict())
        tmp_lr_sched.load_state_dict(self.lr_sched.state_dict())
        tmp_lr_sched_bias.load_state_dict(self.lr_sched_bias.state_dict())

        self.opt = tmp_opt
        self.opt_bias = tmp_opt_bias
        self.lr_sched = tmp_lr_sched
        self.lr_sched_bias = tmp_lr_sched_bias
        """

        # Now that training is done, store the opts etc. in the 'global' attributes
        self.global_opt.load_state_dict(deepcopy(self.opt.state_dict()))
        self.global_opt_bias.load_state_dict(deepcopy(self.opt_bias.state_dict()))
        self.global_lr_sched.load_state_dict(deepcopy(self.lr_sched.state_dict()))
        self.global_lr_sched_bias.load_state_dict(deepcopy(self.lr_sched_bias.state_dict()))

        torch.cuda.empty_cache()
        self.next(self.local_model_validation,  include=base_fedflow_includes + private_includes + ["train_data_size", "test_data_size", "loss", "agg_validation_score", "agg_validation_score_by_label", "collaborator_name"])

    @collaborator
    def local_model_validation(self):
        print(
            (
                "Performing local model validation for collaborator: "
                f"{self.input} in round {self.round_num}"
            )
        )
        print(self.device)
        start_time = time.time()

        print("Test dataset performance")
        self.local_validation_score, self.local_validation_score_by_label, self.test_count_by_label = inference(
            self.model, self.test_loader, self.device
        )
        """
        print("Train dataset performance")
        self.local_validation_score_train, self.local_validation_score_train_by_label, self.train_count_by_label = inference(
            self.model, self._get_round_loader(), self.device
        )
        """
        print(
            (
                "Doing local model validation for collaborator: "
                f"{self.input}: {self.local_validation_score}"
            )
        )
        print(f"local validation time cost {(time.time() - start_time)}")

        self.next(self.join, include=base_fedflow_includes + private_includes + ["train_data_size", "test_data_size", "test_count_by_label", "loss", "agg_validation_score", "agg_validation_score_by_label", "local_validation_score", "local_validation_score_by_label", "collaborator_name"])

    
    @aggregator
    def join(self, inputs):
        """
        ######################################
        # stopping this due to memory usage
        #######################################
        # store individual collaborator results
        for input in inputs:
            # a row for loss
            self.results_dict[self.results_colnames["Round"]].append(self.round_num)
            self.results_dict[self.results_colnames["Loc"]].append(input.input) # col name or "All"
            self.results_dict[self.results_colnames["Lab"]].append("AVE") # label or 'AVE'
            self.results_dict[self.results_colnames["Met"]].append(self.metric_names["Loss"])
            self.results_dict[self.results_colnames["MetVal"]].append(input.loss)
            self.results_dict[self.results_colnames["ModelSeed"]].append(input.model_seed)

            for label in range(10):

                # a row for this collaborator, this label, aggregated model accuracy
                self.results_dict[self.results_colnames["Round"]].append(self.round_num)
                self.results_dict[self.results_colnames["Loc"]].append(input.input) # col name or "All"
                self.results_dict[self.results_colnames["Lab"]].append(label) # label or 'AVE'
                self.results_dict[self.results_colnames["Met"]].append(self.metric_names["AggAcc"])
                self.results_dict[self.results_colnames["MetVal"]].append(input.agg_validation_score_by_label[label])
                self.results_dict[self.results_colnames["ModelSeed"]].append(input.model_seed)


                # a row for this collaborator, this label, local model accuracy
                self.results_dict[self.results_colnames["Round"]].append(self.round_num)
                self.results_dict[self.results_colnames["Loc"]].append(input.input) # col name or "All"
                self.results_dict[self.results_colnames["Lab"]].append(label) # label or 'AVE'
                self.results_dict[self.results_colnames["Met"]].append(self.metric_names["LocAcc"])
                self.results_dict[self.results_colnames["MetVal"]].append(input.local_validation_score_by_label[label])
                self.results_dict[self.results_colnames["ModelSeed"]].append(input.model_seed)
        """

        # To aggregate metrics we need to account for difference in data sizes
        col_weights_train = [input.train_data_size for input in inputs]
        col_weights_test = [input.test_data_size for input in inputs]
        # hard coding for 10 classes
        col_weights_by_label_test = {}
        for label in range(10):
            col_weights_by_label_test[label] = [input.test_count_by_label[label] for input in inputs]

        self.average_loss = np.average([input.loss for input in inputs], weights=col_weights_train)
        self.aggregated_model_accuracy = np.average([input.agg_validation_score for input in inputs], weights=col_weights_test)
        self.aggregated_model_accuracy_by_label = {}
        for label in range(10):
            self.aggregated_model_accuracy_by_label[label] = np.average([input.agg_validation_score_by_label[label] for input in inputs], weights=col_weights_by_label_test[label])
        self.local_model_accuracy = np.average([input.local_validation_score for input in inputs], weights=col_weights_test)
        if self.has_net_ema_val:
            self.aggregated_model_accuracy_ema = np.average([input.agg_validation_score_ema for input in inputs], weights=col_weights_test)

        # from the hlb code
        # The hlb code updates ema on a step basis, but we do so every round
        print(f"\n##############\nRound num: {self.round_num}, self.ema_epoch_start: {self.ema_epoch_start} ema None: {self.net_ema == None}\n#############\n")
        if self.round_num >= self.ema_epoch_start:          
            ## Initialize the ema from the network at this point in time if it does not already exist.... :D
            if self.net_ema is None: # don't snapshot the network yet if so!
                print(f"Initializing ema at round {self.round_num}")
                self.net_ema = NetworkEMA(self.global_model)
            else:
                # We warm up our ema's decay/momentum value over training exponentially according to the hyp config dictionary (this lets us move fast, then average strongly at the end).
                # We use rounds in instead of steps 
                print(f"Updating ema at round {self.round_num}" )
                self.net_ema.update(self.global_model, decay=self.projected_ema_decay_val*(self.round_num/self.total_rounds)**hyp['misc']['ema']['decay_pow'])

        # Storing cross collaborator aggregated results now so that I don't have to know datasizes later
        for label in range(10):

            # a row for this label, AVE of aggregated model accuracies across collaborators
            self.results_dict[self.results_colnames["Round"]].append(self.round_num)
            self.results_dict[self.results_colnames["Loc"]].append("All") # col name or 'All"
            self.results_dict[self.results_colnames["Lab"]].append(label) # label or 'AVE'
            self.results_dict[self.results_colnames["Met"]].append(self.metric_names["AggAcc"])
            self.results_dict[self.results_colnames["MetVal"]].append(self.aggregated_model_accuracy_by_label[label])
            self.results_dict[self.results_colnames["ModelSeed"]].append(self.model_seed)

        # agg accuracy averaged across labels
        self.results_dict[self.results_colnames["Round"]].append(self.round_num)
        self.results_dict[self.results_colnames["Loc"]].append("All") # col name or 'All"
        self.results_dict[self.results_colnames["Lab"]].append('AVE') # label or 'AVE'
        self.results_dict[self.results_colnames["Met"]].append(self.metric_names["AggAcc"])
        self.results_dict[self.results_colnames["MetVal"]].append(self.aggregated_model_accuracy)
        self.results_dict[self.results_colnames["ModelSeed"]].append(self.model_seed)

        # agg loss
        self.results_dict[self.results_colnames["Round"]].append(self.round_num)
        self.results_dict[self.results_colnames["Loc"]].append("All") # col name or 'All"
        self.results_dict[self.results_colnames["Lab"]].append('AVE') # label or 'AVE'
        self.results_dict[self.results_colnames["Met"]].append(self.metric_names["AggLoss"])
        self.results_dict[self.results_colnames["MetVal"]].append(self.average_loss)
        self.results_dict[self.results_colnames["ModelSeed"]].append(self.model_seed)
        
        if self.has_net_ema_val:
            # ema test results also to results
            self.results_dict[self.results_colnames["Round"]].append(self.round_num)
            self.results_dict[self.results_colnames["Loc"]].append("All") # col name or 'All"
            self.results_dict[self.results_colnames["Lab"]].append('AVE') # label or 'AVE'
            self.results_dict[self.results_colnames["Met"]].append(self.metric_names["EMAAcc"])
            self.results_dict[self.results_colnames["MetVal"]].append(self.aggregated_model_accuracy_ema)
            self.results_dict[self.results_colnames["ModelSeed"]].append(self.model_seed)

        
        print("\n####################################################################")
        print(f"Average aggregated model validation values = {self.aggregated_model_accuracy}")
        if self.has_net_ema_val:
            print(f"Average aggregated ema model validation values = {self.aggregated_model_accuracy_ema}")
        print(f"Average training loss = {self.average_loss}")
        print(f"Average local model validation values = {self.local_model_accuracy}")
        print("####################################################################\n")


        # write the results to disk
        if self.round_num == self.total_rounds:
            results_df = pd.DataFrame(self.results_dict)
            results_df.to_csv(self.fpath_results_df, index=False)

        self.model = FedAvg([input.model.cpu() for input in inputs])
        self.global_model.load_state_dict(deepcopy(self.model.state_dict()))

        # NOTE: recall opts and schedulers were set to global at end of train method. We'll take the ones from col 0 to carry forward (as only one can carry forward here)
        self.global_opt = deepcopy(inputs[0].global_opt)
        self.global_opt_bias = deepcopy(inputs[0].global_opt_bias)
        self.global_lr_sched = deepcopy(inputs[0].global_lr_sched)
        self.global_lr_sched_bias = deepcopy(inputs[0].global_lr_sched_bias)

        del inputs
        self.next(self.check_round_completion, include=base_fedflow_includes + additional_roundend_includes)

    @aggregator
    def check_round_completion(self):
        if self.round_num != self.total_rounds:
            if self.aggregated_model_accuracy > self.top_model_accuracy:
                print(
                    (
                        "Accuracy improved to "
                        f"{self.aggregated_model_accuracy} for round {self.round_num}"
                    )
                )
                self.top_model_accuracy = self.aggregated_model_accuracy
            self.round_num += 1
            print(20 * "#")
            print(f"Round {self.round_num}...")
            print(20 * "#")
            self.next(
                self.aggregated_model_validation,
                foreach="collaborators", 
                include=base_fedflow_includes + additional_roundend_includes
            )
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        print(20 * "#")
        print("All rounds completed successfully")
        print(20 * "#")
        print("This is the end of the flow")
        print(20 * "#")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--log_dir",
        type=str,
        default="tutorial_logdir",
        help="Indicate where to save the privacy loss profile and log files during the training",
    )
    argparser.add_argument(
        "--comm_round",
        type=int,
        default=50,
        help="Indicate the communication round of FL",
    )
    argparser.add_argument(
        "--flow_internal_loop_test",
        type=bool,
        default=False,
        help="Indicate enabling of internal loop testing of Federated Flow",
    )
    argparser.add_argument(
        "--optimizer_type",
        type=str,
        default="Adam",
        help="Indicate optimizer to use for training",
    )
    argparser.add_argument(
        "--held_classes",
        type=parse_string_arg,
        default=[],
        help="Classes to hold or supplement with synthetics according to hold_from_cols and synth_to_cols arguments (should be comma separated).",
    )
    argparser.add_argument(
        "--synth_classes",
        type=parse_string_arg,
        default=[],
        help="Classes to hold or supplement with synthetics according to hold_from_cols and synth_to_cols arguments (should be comma separated).",
    )
    argparser.add_argument(
        "--hold_from_cols",
        type=parse_string_arg,
        default=[],
        help="From which collaborators to withhold the target classes (should be comma separated).",
    )
    argparser.add_argument(
        "--synth_to_cols",
        type=parse_string_arg,
        default=[],
        help="To which collaborators to provide synthetic versions of the target class in equal measure to what was pulled (should be comma separated)).",
    )
    argparser.add_argument(
        "--num_cols",
        type=int,
        default=5,
        help="Number of collaborators.",
    )
    argparser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0003,
        help="Learning rate to apply to optimizer",
    )
    argparser.add_argument(
        "--model_seed",
        type=int,
        default=10,
        help="Random seed for model initialization",
    )
    argparser.add_argument(
        "--restoreall_to_one_col",
        type=str,
        default=None,
        help="String for collaborator number to which to restore all of a single class withheld",
    )
    argparser.add_argument(
        "--class_to_restoreall",
        type=str,
        default=None,
        help="String for which class if any to restore all to one collaborator"
    )
    argparser.add_argument(
        "--eps",
        type=str,
        default=None,
        help="String for which epsilon was used for the dp training of the synthetic data generator (or None if DP not used)"
    )
    
    args = argparser.parse_args()

    print(f"\nRunning with args:\n{args}\n\n")

    # Let's always use the integers to track the classes and col numbers (as opposed to strings)
    held_classes = [int(target_class) for target_class in args.held_classes]
    synth_classes = [int(target_class) for target_class in args.synth_classes]
    num_cols = args.num_cols
    learning_rate = args.learning_rate
    hold_from_cols = [int(col_num) for col_num in args.hold_from_cols]
    synth_to_cols = [int(col_num) for col_num in args.synth_to_cols]
    eps = args.eps
    
    restoreall_to_one_col = None
    class_to_restoreall = None
    
    if args.restoreall_to_one_col is not None:
        restoreall_to_one_col = int(args.restoreall_to_one_col)
    if args.class_to_restoreall is not None:
        class_to_restoreall = int(args.class_to_restoreall)

    # set the random seed for repeatable results
    model_seed = args.model_seed
    torch.manual_seed(model_seed)

    # validate certain aspects of the comma separated arguments
    if restoreall_to_one_col is not None:
        if (restoreall_to_one_col in synth_to_cols) and (class_to_restoreall in synth_classes):
            raise ValueError(f"We do not currently support supplementing with synthetics (you've asked for synthetic classes: {synth_classes}) to a collaborator that is getting one of those classes ({class_to_restoreall}) restored completely.")
        if restoreall_to_one_col not in hold_from_cols:
            raise ValueError(f"restore_all_to_one_col needs to be in hold_from_cols")
        if restoreall_to_one_col in synth_to_cols:
            raise ValueError(f"For now, we're avoiding supplmenting a full class with sythetics ... if you change this consider addressing class balance issue.")
        if hold_from_cols != [idx for idx in range(num_cols)]:
            raise ValueError(f"If using restore_to_one_col, this held class must be held from all cols.")

    # there should not be repeat entries in the held and synth classes
    if len(set(held_classes)) != len(held_classes):
        raise ValueError(f"There should not be repeat entries in the held_classes comma separated string argument")
    if len(set(synth_classes)) != len(synth_classes):
        raise ValueError(f"There should not be repeat entries in the synth_classes comma separated string argument")
    
    if not set(hold_from_cols).issubset(set([col_num for col_num in range(num_cols)])):
        raise ValueError(f"hold from cols {hold_from_cols} has entries that do not fit into the range of num_cols provided: {list(range(num_cols))}.")
    # there should not be repeat entries in the hold_from_cols
    if len(set(hold_from_cols)) != len(hold_from_cols):
        raise ValueError(f"There should not be repeat entries in the hold_from_cols comma separated string argument")

    if not set(synth_to_cols).issubset(set([col_num for col_num in range(num_cols)])):
        raise ValueError(f"synth_to_cols {synth_to_cols} has entries that do not fit into the range of num_cols provided: {list(range(num_cols))}.")
    # there should not be repeat entries in the synth_to_cols
    if len(set(synth_to_cols)) != len(synth_to_cols):
        raise ValueError(f"There should not be repeat entries in the synth_to_cols comma separated string argument")

    if not set(synth_to_cols).issubset(set(hold_from_cols)):
        raise ValueError(f"No plan was made to allow for supplementing collaborators that did not have the missing class")

    #######################################
    # Hard coded params
    ######################################

    # some hard coded paths
    if eps is None:
        eps_string = 'None'
    else:
        eps_string = eps
    module_path = os.path.dirname(os.path.realpath(__file__))

    if eps_string != 'None':
        module_path = os.path.join(module_path, f'DP_RESULTS_PARDIR_eps_{eps_string}')
    fpath_data_by_col = os.path.join(module_path, 'data', 'by_collaborator', f'data_by_col_holding_{list_to_string(held_classes)}_from_{list_to_string(hold_from_cols)}_supplementing_{list_to_string(synth_to_cols)}_num_cols_{num_cols}_with_synth_classes_{list_to_string(synth_classes)}_restoreall_{class_to_restoreall}_to_col_{restoreall_to_one_col}_HLBprep.npy')
    fpath_results_df = os.path.join(module_path, 'Results', get_results_path(module_path=module_path, 
                                    held_classes=held_classes, 
                                    synth_classes=synth_classes, 
                                    hold_from_cols=hold_from_cols, 
                                    synth_to_cols= synth_to_cols, 
                                    LearningRate=learning_rate, 
                                    num_cols=num_cols, 
                                    restoreall_to_one_col=restoreall_to_one_col, 
                                    class_to_restoreall=class_to_restoreall, 
                                    ModelSeed=model_seed, 
                                    waugs=True))

    # this was old class conditional one 
    # fpaths_synthetic_data = {target_class: f"/home/edwardsb/repositories/nvidia_edm/class_{target_class}_batchsize_64_266_batches.pkl"for target_class in synth_classes}
    if eps_string == 'None':
         fpaths_synthetic_data = {6: '/raid/edwardsb/projects/fl_with_diffusion/training_outdir/00000-data_for_frog_model-uncond-ddpmpp-edm-gpus8-batch512-fp32/samples_iter_0_network-snapshot-005018.pkl'}
    else:
        print(f"\n\n##### USING DP data with epsilon: {eps_string}\n\n")
        fpaths_synthetic_data = {6: f'/raid/edwardsb/projects/fl_with_diffusion/dp_model_sample_arrays/eps_{eps_string}_17024_3_32_32.pkl'}
    # some other hard coded
    shuffle_seed = 1234567

    # Setup participants
    # If running with GPU and 1 GPU is available then
    # Set `num_gpus=0.3` to run on GPU
    aggregator = Aggregator()

    collaborator_names = [f"Col_{col_num}" for col_num in range(num_cols)]

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Download and setup the train, and test dataset
    transform = transforms.Compose([transforms.ToTensor()])

    cifar_train = CIFAR10(root="./data", train=True, download=True, transform=transform)

    cifar_test = CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Split the dataset in train, test
    N_total_samples = len(cifar_test) + len(cifar_train)

    # now load up the sythetic supplement data from which we will sample to augmented collaborators (synth_to_cols)
    synthetic_data = {}
    if len(synth_to_cols) != 0:
        for target_class, fpath in fpaths_synthetic_data.items():
            with open(fpath, 'rb') as _file:
                X_supp, Y_supp = pickle.load(_file)
                print(f"Loding up synthetic data for class {target_class}")
                print(f"Features have range: [{np.amin(X_supp)}, {np.amax(X_supp)}] and shape: {X_supp.shape}")
                print(f"Synthetic labels have values: {np.unique(Y_supp)} and shape {Y_supp.shape}\n")
                print(f"Feature type is: {X_supp.dtype} and Label type is {Y_supp.dtype}")
                print(f"We will use this to supplement cols {synth_to_cols}.\n\n")
                synthetic_data[target_class] = (X_supp, Y_supp)

    train_dataset = cifar_train
    train_dataset.targets = np.array(train_dataset.targets)
    test_dataset = cifar_test
    test_dataset.targets = np.array(test_dataset.targets)

    print(
        (
            f"Pre-split dataset info (total {N_total_samples}): "
            f"train - {len(train_dataset)}, "
            f"test - {len(test_dataset)}, \n"
            f"Supplement sizes: {[f'Class {target_class} has {len(X_supp)} samples' for target_class, (X_supp, Y_supp) in synthetic_data.items()]}"
        )
    )

 
    # Split train, test datasets among collaborators
    
    # First check whether the results are on disk
    if os.path.exists(fpath_data_by_col):
        # raise ValueError(f"For now dissabling precomputed data as I want to not accidentally use old data.")
        print(f"\nLoading by col data from: {fpath_data_by_col}...\n")
        train_data_by_col, test_data_by_col = np.load(fpath_data_by_col, allow_pickle=True)
    else:
        print(f"\nConstructing by col data...\n")
        train_dict = features_labels_to_dict(features=train_dataset.data, labels=train_dataset.targets)
        test_dict = features_labels_to_dict(features=test_dataset.data, labels=test_dataset.targets)
        
        
        print(f"Organizing train data by class.")
        train_data_by_class = split_data_by_class(_dict=train_dict)
        print(f"Organizing test data by class.")
        test_data_by_class = split_data_by_class(_dict=test_dict)
        
        
        # separate target classes in train data to distribute as designated
        if len(held_classes) != 0:
            initial_held_train_by_class, left_over_train_by_class = split_off_classes(target_classes=held_classes, dict_by_class=train_data_by_class)
        else:
            left_over_train_by_class = train_data_by_class
            initial_held_train_by_class = None

        # split the test data
        print(f"Splitting test data in a stratified manor.")
        test_data_by_col, _ = stratified_split(dict_by_class=test_data_by_class, n_parts=num_cols, shuffle=False, shuffle_seed=shuffle_seed)

        # split the train data (keeping target classes separate which we'll use or not depending on designation)
        print(f"Splitting left over train data (after holding held_classes) in a stratified manor.")
        train_data_by_col, _ = stratified_split(dict_by_class=left_over_train_by_class, n_parts=num_cols, shuffle=True, shuffle_seed=shuffle_seed)
        # now splitting the initially held classes (may put some back accoring to entries in hold_from_cols)
        if initial_held_train_by_class is not None:
            if restoreall_to_one_col is not None:
                # here the class to resore all is handled separately
                print(f"\nPerforming stratified split, but preparing to restore all.\n")
                class_to_restoreall_by_split, class_to_restoreall_counts_by_split = stratified_split(dict_by_class={class_to_restoreall: initial_held_train_by_class[class_to_restoreall]}, 
                                                                                                   n_parts=num_cols, 
                                                                                                   shuffle=True, 
                                                                                                   shuffle_seed=shuffle_seed)
                
                initially_held_other_than_class_to_restoreall = {_class: initial_held_train_by_class[_class] for _class in initial_held_train_by_class if (_class != class_to_restoreall)}

                if initially_held_other_than_class_to_restoreall != {}:
                    other_than_restoreall_class_by_col, other_than_restoreall_class_counts_by_split = stratified_split(dict_by_class=initially_held_other_than_class_to_restoreall, 
                                                                                                n_parts=num_cols, 
                                                                                                shuffle=True, 
                                                                                                shuffle_seed=shuffle_seed)
                else:
                    other_than_restoreall_class_by_col = None

                # here we may supplement also, and we want to be able to keep track of supplement class counts in one dictionary
                target_counts = class_to_restoreall_counts_by_split
                if other_than_restoreall_class_by_col is not None:
                    target_counts.update(other_than_restoreall_class_counts_by_split)
            else:
                print(f"\nPerforming stratified split and will not be restoring all to any.\n")
                initial_held_train_by_col, target_counts_by_class_by_split = stratified_split(dict_by_class=initial_held_train_by_class, 
                                                                                              n_parts=num_cols, 
                                                                                              shuffle=True, 
                                                                                              shuffle_seed=shuffle_seed)
                target_counts = target_counts_by_class_by_split 
        else:
            print(f"\nNo classes held and so not requiring to stratified split any held data.\n")
        

        # keep track of which synthetic samples have already been used
        offset_target_data_idx_by_class = {target_class: 0 for target_class in synthetic_data}
        # synthetic replacement falls only under this first conditional as we assume supplmenting only occurs when the classes have beeen held
        if initial_held_train_by_class is not None:
            if restoreall_to_one_col is not None:
                # here one class (which must belong to held_classes) must be listed as held from all cols (this is enforced above) and is completely restored to one in the form of num_cols different loaders (to maintane class balance in each)
                for col_num in range(num_cols):
                    if col_num == restoreall_to_one_col:
                        # This collaborator is known to be in hold_from_cols, and so we are not restoring any of the held classes (except for the class_to_restoreall) (hence comment out below)
                        # train_data_by_col[col_num] = combine_dicts(*[train_data_by_col[col_num], other_than_restoreall_class_by_col[col_num]], shuffle=True, shuffle_seed=shuffle_seed)
                        # now create num_col loaders each that holds a shard of the restoreall class
                        train_data_by_col[col_num] = [combine_dicts(*[train_data_by_col[col_num], class_to_restoreall_by_split[other_col_num]], shuffle=True, shuffle_seed=shuffle_seed) for other_col_num in range(num_cols)]
                    elif col_num not in hold_from_cols:
                        if restoreall_to_one_col is not None:
                            raise ValueError(f"The only current use of restoreall_to_one_col is for all other collaborators to be in hold_from_cols")
                        # here they do not get the restoreall class but may get other classes back if they are not in hold_from_cols (an assumption that is enforced above against what is designated in the hold_from_cols)
                        if other_than_restoreall_class_by_col is not None:
                            train_data_by_col[col_num] = combine_dicts(*[train_data_by_col[col_num], other_than_restoreall_class_by_col[col_num]], shuffle=True, shuffle_seed=shuffle_seed)
                    
                     
            else:
                for col_num in range(num_cols):
                    if col_num not in hold_from_cols:
                        # here replace the  held data shard (so only the portion of that class in size of an equal split among all cols)
                        train_data_by_col[col_num] = combine_dicts(*[train_data_by_col[col_num], initial_held_train_by_col[col_num]], shuffle=True, shuffle_seed=shuffle_seed)
                    
            # now supplement if needed        
            for col_num in range(num_cols):  
                if col_num in synth_to_cols:
                    # recall we only currently supplement in the case that classes have been held
                    # here is where we assume only cols who get held from will be supplemented (will restore sythetics in same count as real were pulled)
                    supp_images = None
                    supp_labels = None
                    for target_class in synth_classes:
                        X_supp, Y_supp = synthetic_data[target_class]
                        start = offset_target_data_idx_by_class[target_class]
                        end = start + target_counts[target_class][col_num]
                        offset_target_data_idx_by_class[target_class] = end

                        if end > len(X_supp):
                            raise ValueError(f"Trying to pull off more than {end} samples from synthetic data and don't have them ... we are supplementing cols: {synth_to_cols} and only have {len(X_supp)} sythetic samples for this class: {target_class}")
                        
                        if not supp_images:
                            supp_images = X_supp[start:end]
                            supp_labels = Y_supp[start:end]
                        else:
                            supp_images = np.concatenate([supp_images, X_supp[start:end]], axis=0)
                            supp_labels = np.concatenate([supp_labels, Y_supp[start:end]], axis=0)                       

                    # if we have restored all to one col then train_data_by_col is a list
                    if isinstance(train_data_by_col[col_num], list):
                        # restoreall col is assumed to not be suppemented
                        raise ValueError(f"Restorall class is assumed to not be suppmented ... something is wrong.")
                    else:
                        train_data_by_col[col_num] = combine_dicts(train_data_by_col[col_num], features_labels_to_dict(features=supp_images,labels=supp_labels), shuffle=True, shuffle_seed=shuffle_seed)
                # Now perform hlb preprocessing

        ######### some data preprocessing ##################
        for col_num in range(num_cols):
            col_train_feats, col_train_labels = dict_to_features_labels(train_data_by_col[col_num])
            col_test_feats, col_test_labels = dict_to_features_labels(test_data_by_col[col_num])

            # convert to torch tensors
            col_train_feats = torch.tensor(col_train_feats, dtype=torch.float32)
            col_test_feats = torch.tensor(col_test_feats, dtype=torch.float32)
            col_train_labels = torch.tensor(col_train_labels, dtype=torch.int64)
            col_test_labels = torch.tensor(col_test_labels, dtype=torch.int64)




            col_data_train_std, col_data_train_mean = torch.std_mean(col_train_feats, dim=(0, 2, 3)) # dynamically calculate the std and mean from the data. this shortens the code and should help us adapt to new datasets!

            def batch_normalize_images(input_images, mean, std):
                return (input_images - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

            # preload with our mean and std
            batch_normalize_images = partial(batch_normalize_images, mean=col_data_train_mean, std=col_data_train_std)

            ## Batch normalize datasets, as well as convert dataset to FP16 now for the rest of the process....
            col_train_feats = batch_normalize_images(col_train_feats).half().requires_grad_(False)
            col_test_feats  = batch_normalize_images(col_test_feats).half().requires_grad_(False)

            # NOT PADDING FOR NOW, GOT AN ERROR SINCE I HAD HALF PRECISION HERE AND F.pad it says is not defined for that
            if DONE_Brandon_DEBUG: 
                # Pad the GPU training dataset
                if hyp['net']['pad_amount'] > 0:
                    ## Uncomfortable shorthand, but basically we pad evenly on all _4_ sides with the pad_amount specified in the original dictionary
                    col_train_feats = F.pad(col_train_feats, (hyp['net']['pad_amount'],)*4, 'reflect')
            
            # Convert this to one-hot to support the usage of cutmix (or whatever strange label tricks/magic you desire!)
            col_train_labels = F.one_hot(col_train_labels).half()
            col_test_labels = F.one_hot(col_test_labels).half()

            train_data_by_col[col_num] = features_labels_to_dict(features=col_train_feats, labels=col_train_labels)
            test_data_by_col[col_num] = features_labels_to_dict(features=col_test_feats, labels=col_test_labels)
        
        ##############################################################


        np.save(fpath_data_by_col, (train_data_by_col, test_data_by_col))
    if isinstance(train_data_by_col[0], list):
        print(f"#############################################")
        print(f"Train data by col sizes (we've restored all to collaborator 0 so that the first one is a list):")
        print(f"In this case train_data_by_col[0] is of type: {type(train_data_by_col[0])}")

    else:
        print(f"#############################################")
        print(f"Train data by col sizes:")
        print(f"{[len(train_data_by_col[col_num]['features']) for col_num in train_data_by_col]}\n")

    print(f"Test data by col sizes:")
    print(f"{[len(test_data_by_col[col_num]['features']) for col_num in test_data_by_col]}\n")

    print(f"#############################################")

    # this function will be called before executing collaborator steps
    # which will return private attributes dictionary for each collaborator
    def callable_to_initialize_collaborator_private_attributes(index, 
                                                               train_data_by_col, 
                                                               test_data_by_col, 
                                                               train_ds, 
                                                               test_ds,
                                                               opts_and_scheds, 
                                                               collaborator_names, 
                                                               args):
        
        local_test = deepcopy(test_ds)
        local_test.data = test_data_by_col[index]['features']
        local_test.targets = test_data_by_col[index]['labels']

        # if we have restored all of a class to one collaborator, then the local_train is a list (will create num_col loaders)
        if isinstance(train_data_by_col[index], list):
            # construct the training and test and population dataset
            local_trains = [deepcopy(train_ds) for idx in range(num_cols)]

            # here we prepare to create num_cols different train loaders (test loaders same), each of which is identical in all classes excecpt
            # in the class, class_to_restoreall in which case it holds a different shard to be used round robin over training rounds
            for loader_idx in range(num_cols):
                local_trains[loader_idx].data = train_data_by_col[index][loader_idx]['features']
                local_trains[loader_idx].targets = train_data_by_col[index][loader_idx]['labels']

            return \
                {
                "train_loader": [torch.utils.data.DataLoader(local_trains[loader_idx], 
                                                             batch_size=batchsize, 
                                                             shuffle=True
                                                            ) 
                                for loader_idx in range(num_cols)],
                "test_loader": torch.utils.data.DataLoader(local_test, 
                                                           batch_size=batchsize_test, 
                                                           shuffle=False
                                                          ),
                "opt": opts_and_scheds[collaborator_names[index]]['opt'], 
                "opt_bias": opts_and_scheds[collaborator_names[index]]['opt_bias'],
                "lr_sched": opts_and_scheds[collaborator_names[index]]['lr_sched'],
                "lr_sched_bias": opts_and_scheds[collaborator_names[index]]['lr_sched_bias']
                }
        else:
            local_train = deepcopy(train_ds)
            local_train.data = train_data_by_col[index]['features']
            local_train.targets = train_data_by_col[index]['labels']           
            
            return {
                    "train_loader": torch.utils.data.DataLoader(local_train, 
                                                                batch_size=batchsize, 
                                                                shuffle=True
                                                                ),
                    "test_loader": torch.utils.data.DataLoader(local_test, 
                                                               batch_size=batchsize_test, 
                                                               shuffle=False
                                                                ),
                "opt": opts_and_scheds[collaborator_names[index]]['opt'], 
                "opt_bias": opts_and_scheds[collaborator_names[index]]['opt_bias'],
                "lr_sched": opts_and_scheds[collaborator_names[index]]['lr_sched'],
                "lr_sched_bias": opts_and_scheds[collaborator_names[index]]['lr_sched_bias']
                    }
    
    model =  make_net()
        
    opts_and_scheds = {}
    for col_idx, collab_name in enumerate(collaborator_names):
        opt, opt_bias, lr_sched, lr_sched_bias = get_optimizers_and_schedulers(model=model, 
                                                                           num_train_samples=len(train_data_by_col[col_idx]['features']), 
                                                                           batchsize=batchsize, 
                                                                           total_rounds=args.comm_round, 
                                                                           hyp=hyp)
        opts_and_scheds[collab_name] = {
            "opt": opt,
            "opt_bias": opt_bias,
            "lr_sched": lr_sched,
            "lr_sched_bias": lr_sched_bias
        }

    collaborators = []
    for idx, collab_name in enumerate(collaborator_names):
        collaborators.append(
            Collaborator(
                name=collab_name,
                private_attributes_callable=callable_to_initialize_collaborator_private_attributes,
                # If 1 GPU is available in the machine
                # Set `num_gpus=0.0` to `num_gpus=0.3` to run on GPU
                # with ray backend with 2 collaborators
                num_cpus=0.0,
                num_gpus=0.0,
                index=idx,
                collaborator_names=collaborator_names,
                train_data_by_col=train_data_by_col,
                test_data_by_col=test_data_by_col,
                train_ds=train_dataset, 
                test_ds=test_dataset, 
                opts_and_scheds=opts_and_scheds,
                args=args,
            )
        )

    # Set backend='ray' to use ray-backend
    local_runtime = LocalRuntime(
        aggregator=aggregator, collaborators=collaborators, backend="single_process"
    )

    print(f"Local runtime collaborators = {local_runtime.collaborators}")

    # change to the internal flow loop
    top_model_accuracy = 0
    
    flflow = FederatedFlow(
        model_seed=model_seed,
        model=model,
        device=device,
        total_rounds=args.comm_round,
        top_model_accuracy=top_model_accuracy,
        flow_internal_loop_test=args.flow_internal_loop_test,
        fpath_results_df=fpath_results_df
    )

    flflow.runtime = local_runtime
    flflow.run()
