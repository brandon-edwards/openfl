# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# copied and modified by Brandon Edwards from https://github.com/brandon-edwards/openfl/blob/develop/openfl-tutorials/experimental/Privacy_Meter/cifar10_PM.py
# -----------------------------------------------------------
# Primary author: Hongyan Chang <hongyan.chang@intel.com>
# Co-authored-by: Anindya S. Paul <anindya.s.paul@intel.com>
# Co-authored-by: Brandon Edwards <brandon.edwards@intel.com>
# ------------------------------------------------------------

# NOTE: Version 2 has the following changes:
#    Now I will use the natural CIFAR10 train/test split and that test data (for compute reasons only) will be split equally between cols 0, 1, and 2 (though make num cols variable)
#    I will perform 3 different approaches (differeing in how train data looks across cols): 
#         - normal federation for baseline (stratified split)
#         - col 0 has real frogs and cols 1 & 2 have synthetic frogs
#         - cols 0, 1, and 2 all have synthetic frogs

from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision.models import convnext_base

import numpy as np
from functools import partial

from openfl.experimental.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.runtime import LocalRuntime
from openfl.experimental.placement import aggregator, collaborator
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
from data_utils import split_data_by_class, stratified_split, combine_dicts, features_labels_to_dict, split_off_classes

warnings.filterwarnings("ignore")

batch_size_train = 1024
batch_size_test = 1024
# changed learning rate to come in via arguments
# learning_rate = 0.2
momentum = 0.9
log_interval = 10


# we will use the convnext model
Net = partial(convnext_base, num_classes=10)

def default_optimizer(model, learning_rate, optimizer_type=None, optimizer_like=None):
    """
    Return a new optimizer based on the optimizer_type or the optimizer template

    Args:
        model:   NN model architected from nn.module class
        learning_rate: apply to optimizer
        optimizer_type: "SGD" or "Adam"
        optimizer_like: "torch.optim.SGD" or "torch.optim.Adam" optimizer
    """
    if optimizer_type == "SGD" or isinstance(optimizer_like, optim.SGD):
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_type == "Adam" or isinstance(optimizer_like, optim.Adam):
        return optim.Adam(model.parameters())


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
    test_loss = 0
    correct = 0
    test_loss_by_label = {label: 0 for label in range(10)}
    correct_by_label = {label: 0 for label in range(10)}
    count_by_label = {label: 0 for label in range(10)}
    with torch.no_grad():
        for batch_data, batch_target in test_loader:
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
                    criterion = nn.CrossEntropyLoss()
                    test_loss += criterion(output, target).item()
                    test_loss_by_label[label] += criterion(output, target).item()
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
    test_loss /= len(test_loader)
    for label in test_loss_by_label:
        if count_by_label[label] != 0:
            test_loss_by_label[label] /= count_by_label[label]
        else:
            test_loss_by_label[label] = 0
    print(f"\nBrandon DEBUG - correct_by_label: {correct_by_label}, count_by_label: {count_by_label}\n\n")
    accuracy_by_label = {label: (float(correct_by_label[label] / count_by_label[label]) if count_by_label[label] != 0 else 1.0) for label in count_by_label }

    accuracy = float(correct / len(test_loader.dataset))
    print(
        (
            f"Test set: Avg. loss: {test_loss}, "
            f"Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * accuracy}%)\n"
        )
    )
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


def load_previous_round_model_and_optimizer_and_perform_testing(
    model, global_model, optimizer, learning_rate, collaborator_name, round_num, device
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
        optimizer: local collaborator optimizer at the current round
        learning_rate: apply to optimizer
        collaborator_name: name of the collaborator (Type:string)
        round_num: current round (Type:int)
        device: CUDA device id or "cpu"
    """
    print(f"Loading model and optimizer state dict for round {round_num-1}")
    model_prevround = Net()  # instantiate a new model
    model_prevround = model_prevround.to(device)
    optimizer_prevround = default_optimizer(model=model_prevround, learning_rate=learning_rate, optimizer_like=optimizer)
    if os.path.isfile(
        f"Collaborator_{collaborator_name}_model_config_roundnumber_{round_num-1}.pickle"
    ):
        with open(
            f"Collaborator_{collaborator_name}_model_config_roundnumber_{round_num-1}.pickle",
            "rb",
        ) as f:
            model_prevround_config = pickle.load(f)
            model_prevround.load_state_dict(model_prevround_config["model_state_dict"])
            optimizer_prevround.load_state_dict(
                model_prevround_config["optim_state_dict"]
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

                if isinstance(optimizer, optim.SGD):
                    if optimizer.state_dict()["state"] != {}:
                        for param_idx in optimizer.state_dict()["param_groups"][0][
                            "params"
                        ]:
                            for tensor_1, tensor_2 in zip(
                                optimizer.state_dict()["state"][param_idx][
                                    "momentum_buffer"
                                ],
                                optimizer_prevround.state_dict()["state"][param_idx][
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
                for idx, param in enumerate(optimizer.param_groups[0]["params"]):
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
    model, optimizer, collaborator_name, round_num
):
    """
    Save the model and optimizer state dictionary
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
        optimizer: local collaborator optimizer at the current round
        collaborator_name: name of the collaborator (Type:string)
        round_num: current round (Type:int)
    """
    model_config = {
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
    }
    with open(
        f"Collaborator_{collaborator_name}_model_config_roundnumber_{round_num}.pickle",
        "wb",
    ) as f:
        pickle.dump(model_config, f)


class FederatedFlow(FLSpec):
    def __init__(
        self,
        model_seed,
        model,
        optimizers,
        learning_rate,
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
        self.global_model = Net()
        self.optimizers = optimizers
        self.learning_rate = learning_rate
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
                             "LocAcc": "Local Model Accuracy"}
        self.results_dict = {name: [] for name in self.results_colnames.values()}
        print(20 * "#")
        print(f"Round {self.round_num}...")
        print(20 * "#")

    @aggregator
    def start(self):
        self.start_time = time.time()
        print("Performing initialization for model")
        self.collaborators = self.runtime.collaborators
        self.private = 10
        self.next(
            self.aggregated_model_validation,
            foreach="collaborators",
            exclude=["private"],
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
        print(f"{self.input} value of {self.agg_validation_score} and by label: {self.agg_validation_score_by_label}")
        self.collaborator_name = self.input
        self.next(self.train)

    @collaborator
    def train(self):
        print(20 * "#")
        print(
            f"Performing model training for collaborator {self.input} in round {self.round_num}"
        )

        # store data size for later use (currently allowing these to get overwritten repeatedly)
        self.train_data_size = len(self.train_loader)
        self.test_data_size = len(self.test_loader)

        self.model.to(self.device)
        self.optimizer = default_optimizer(
            model=self.model, optimizer_like=self.optimizers[self.input], learning_rate=self.learning_rate
        )

        if self.round_num > 0:
            self.optimizer.load_state_dict(
                deepcopy(self.optimizers[self.input].state_dict())
            )
            optimizer_to_device(optimizer=self.optimizer, device=self.device)

            if self.flow_internal_loop_test:
                load_previous_round_model_and_optimizer_and_perform_testing(
                    model=self.model,
                    global_model=self.global_model,
                    optimizer=self.optimizer,
                    learning_rate=self.learning_rate,
                    collaborator_name=self.collaborator_name,
                    round_num=self.round_num,
                    device=self.device,
                )

        self.model.train()
        train_losses = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target).to(self.device)
            loss.backward()
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                train_losses.append(loss.item())

        self.loss = np.mean(train_losses)
        self.training_completed = True

        if self.flow_internal_loop_test:
            save_current_round_model_and_optimizer_for_next_round_testing(
                self.model, self.optimizer, self.collaborator_name, self.round_num
            )

        self.model.to("cpu")
        tmp_opt = deepcopy(self.optimizers[self.input])
        tmp_opt.load_state_dict(self.optimizer.state_dict())
        self.optimizer = tmp_opt
        torch.cuda.empty_cache()
        self.next(self.local_model_validation)

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
        print("Train dataset performance")
        self.local_validation_score_train, self.local_validation_score_train_by_label, self.train_count_by_label = inference(
            self.model, self.train_loader, self.device
        )

        print(
            (
                "Doing local model validation for collaborator: "
                f"{self.input}: {self.local_validation_score}"
            )
        )
        print(f"local validation time cost {(time.time() - start_time)}")

        self.next(self.join, exclude=["training_completed"])

    
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


        # Storing cross collaborator aggregated results now so that I don't have to know datasizes later

        for label in range(10):

            # a row for this label, AVE of aggregated model accuracies across collaborators
            self.results_dict[self.results_colnames["Round"]].append(self.round_num)
            self.results_dict[self.results_colnames["Loc"]].append("All") # col name or 'All"
            self.results_dict[self.results_colnames["Lab"]].append(label) # label or 'AVE'
            self.results_dict[self.results_colnames["Met"]].append(self.metric_names["AggAcc"])
            self.results_dict[self.results_colnames["MetVal"]].append(self.aggregated_model_accuracy_by_label[label])
            self.results_dict[self.results_colnames["ModelSeed"]].append(self.model_seed)

        # same averaged across labels
        self.results_dict[self.results_colnames["Round"]].append(self.round_num)
        self.results_dict[self.results_colnames["Loc"]].append("All") # col name or 'All"
        self.results_dict[self.results_colnames["Lab"]].append('AVE') # label or 'AVE'
        self.results_dict[self.results_colnames["Met"]].append(self.metric_names["AggAcc"])
        self.results_dict[self.results_colnames["MetVal"]].append(self.aggregated_model_accuracy)
        self.results_dict[self.results_colnames["ModelSeed"]].append(self.model_seed)

        
        print("\n####################################################################")
        print(f"Average aggregated model validation values = {self.aggregated_model_accuracy}")
        print(f"Average training loss = {self.average_loss}")
        print(f"Average local model validation values = {self.local_model_accuracy}")
        print("####################################################################\n")

        # write the results to disk
        if self.round_num == self.total_rounds:
            results_df = pd.DataFrame(self.results_dict)
            results_df.to_csv(self.fpath_results_df, index=False)

        self.model = FedAvg([input.model.cpu() for input in inputs])
        self.global_model.load_state_dict(deepcopy(self.model.state_dict()))
        self.optimizers.update(
            {input.collaborator_name: input.optimizer for input in inputs}
        )

        del inputs
        self.next(self.check_round_completion)

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
                exclude=["private"],
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
    
    args = argparser.parse_args()
    held_classes = [int(target_class) for target_class in args.held_classes]
    synth_classes = [int(target_class) for target_class in args.synth_classes]
    num_cols = args.num_cols
    learning_rate = args.learning_rate
    hold_from_cols = [int(col_num) for col_num in args.hold_from_cols]
    synth_to_cols = [int(col_num) for col_num in args.synth_to_cols]

    # set the random seed for repeatable results
    model_seed = args.model_seed
    torch.manual_seed(model_seed)

    # validate certain aspects of the comma separated arguments

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
    module_path = os.path.dirname(os.path.realpath(__file__))
    fpath_data_by_col = os.path.join(module_path, 'data', 'by_collaborator', f'data_by_col_holding_{list_to_string(held_classes)}_from_{list_to_string(hold_from_cols)}_supplementing_{list_to_string(synth_to_cols)}_num_cols_{num_cols}_with_synth_classes_{list_to_string(synth_classes)}.npy')
    fpath_results_df = os.path.join(module_path, 'Results', f'v2fed_results_missing_{list_to_string(held_classes)}_holding_from_{list_to_string(hold_from_cols)}_supplementing_{list_to_string(synth_to_cols)}_with_synth_classes_{list_to_string(synth_classes)}_lr_{learning_rate}_num_cols_{num_cols}_model_seed_{model_seed}.csv')
    fpaths_synthetic_data = {target_class: f"/home/edwardsb/repositories/nvidia_edm/class_{target_class}_batchsize_64_266_batches.pkl"for target_class in synth_classes}
    
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
        train_data_by_col, test_data_by_col = np.load(fpath_data_by_col, allow_pickle=True)
    else:
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
        if initial_held_train_by_class:
            initial_held_train_by_col, target_counts_by_class_by_split = stratified_split(dict_by_class=initial_held_train_by_class, n_parts=num_cols, shuffle=True, shuffle_seed=shuffle_seed)
        else:
            initial_held_train_by_col = None
            target_counts_by_class = {idx: 0 for idx in range(10)}
        

        # keep track of which synthetic samples have already been used
        offset_target_data_idx_by_class = {target_class: 0 for target_class in synthetic_data}
        if initial_held_train_by_col:
            for col_num in initial_held_train_by_col:
                if col_num not in hold_from_cols:
                    # put back the target classes
                    train_data_by_col[col_num] = combine_dicts(*[train_data_by_col[col_num], initial_held_train_by_col[col_num]], shuffle=True, shuffle_seed=shuffle_seed)
                elif col_num in synth_to_cols:
                    # here is where we assume only cols who get held from will be supplemented (will restore sythetics in same count as real were pulled)
                    supp_images = None
                    supp_labels = None
                    for target_class in synth_classes:
                        X_supp, Y_supp = synthetic_data[target_class]

                        start = offset_target_data_idx_by_class[target_class]
                        end = start + target_counts_by_class_by_split[target_class][col_num]
                        offset_target_data_idx_by_class[target_class] = end

                        if end > len(X_supp):
                            raise ValueError(f"Trying to pull off more than {end} samples from synthetic data and don't have them ... we are supplementing cols: {synth_to_cols} and only have {len(X_supp)} sythetic samples for this class: {target_class}")
                        
                        if not supp_images:
                            supp_images = X_supp[start:end]
                            supp_labels = Y_supp[start:end]
                        else:
                            supp_images = np.concatenate([supp_images, X_supp[start:end]], axis=0)
                            supp_labels = np.concatenate([supp_labels, Y_supp[start:end]], axis=0)                       

                    train_data_by_col[col_num] = combine_dicts(train_data_by_col[col_num], features_labels_to_dict(features=supp_images,labels=supp_labels), shuffle=True, shuffle_seed=shuffle_seed)

        np.save(fpath_data_by_col, (train_data_by_col, test_data_by_col))

    print(f"#############################################")
    print(f"Train data by col sizes:")
    print(f"{[len(train_data_by_col[col_num]['features']) for col_num in train_data_by_col]}\n")

    print(f"Test data by col sizes:")
    print(f"{[len(test_data_by_col[col_num]['features']) for col_num in test_data_by_col]}\n")

    print(f"#############################################")

    # this function will be called before executing collaborator steps
    # which will return private attributes dictionary for each collaborator
    def callable_to_initialize_collaborator_private_attributes(
        index, train_data_by_col, test_data_by_col, train_ds, test_ds, n_collaborators, args
    ):
        # construct the training and test and population dataset
        local_train = deepcopy(train_ds)
        local_test = deepcopy(test_ds)

        local_train.data = train_data_by_col[index]['features']
        local_train.targets = train_data_by_col[index]['labels']

        local_test.data = test_data_by_col[index]['features']
        local_test.targets = test_data_by_col[index]['labels']            
        
        return {
            "train_dataset": local_train,
            "test_dataset": local_test,
            "train_loader": torch.utils.data.DataLoader(
                local_train, batch_size=batch_size_train, shuffle=True
            ),
            "test_loader": torch.utils.data.DataLoader(
                local_test, batch_size=batch_size_test, shuffle=False
            ),
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
                n_collaborators=len(collaborator_names),
                train_data_by_col=train_data_by_col,
                test_data_by_col=test_data_by_col,
                train_ds=train_dataset, 
                test_ds=test_dataset,
                args=args,
            )
        )

    # Set backend='ray' to use ray-backend
    local_runtime = LocalRuntime(
        aggregator=aggregator, collaborators=collaborators, backend="single_process"
    )

    print(f"Local runtime collaborators = {local_runtime.collaborators}")

    # change to the internal flow loop
    model = Net()
    top_model_accuracy = 0
    optimizers = {
        collaborator.name: default_optimizer(model, optimizer_type=args.optimizer_type, learning_rate=learning_rate)
        for collaborator in collaborators
    }
    flflow = FederatedFlow(
        model_seed=model_seed,
        model=model,
        optimizers=optimizers,
        device=device,
        total_rounds=args.comm_round,
        top_model_accuracy=top_model_accuracy,
        flow_internal_loop_test=args.flow_internal_loop_test,
        fpath_results_df=fpath_results_df, 
        learning_rate=learning_rate
    )

    flflow.runtime = local_runtime
    flflow.run()
