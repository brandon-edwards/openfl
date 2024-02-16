# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Contributors: Micah Sheller, Patrick Foley, Brandon Edwards  - DELETEME?

"""
# TODO: Clean up imports

import os
import subprocess

from .nnunet_v1 import train_nnunet

import numpy as np
import shutil
import time

import torch

from openfl.utilities import TensorKey
from .runner import TaskRunner
from .external_train_functions import train_mnist_net, load_json
from .runner_pt_utils import rebuild_model_util, _derive_opt_state_dict, expand_derived_opt_state_dict
from .runner_pt_utils import initialize_tensorkeys_for_functions_util, to_cpu_numpy
from .runner_pt_utils import _derive_opt_state_dict, expand_derived_opt_state_dict

def get_train_function():
    return NotImplementedError()


class WeightsOnlyPyTorchCheckpointTaskRunner(TaskRunner):
    """An abstract class for PyTorch model based Tasks, where training, validation etc. are processes that
       pull model state from a PyTorch checkpoint."""

    def __init__(self,
                 checkpoint_out_path = None,
                 checkpoint_in_path = None,
                 device = 'cuda:0',
                 config_path=None,
                 **kwargs):
        """Initialize.

        Args:
            checkpoint_out_path(str)    : Path to the model checkpoint that will be used to start local training.
            checkpoint_in_path(str)     : Path to model checkpoint that results from performing local training.
            device(str)                 : Device ('cpu' or 'cuda') to be used for training and validation script computations.
            kwargs                      : Additional key work arguments (will be passed to rebuild_model, initialize_tensor_key_functions, TODO: <Fill this in>).
            config_path(str)            : Path to the configuration file used by the training and validation script.
            TODO: 
        """ 
        super().__init__(**kwargs)

        # TODO: Both 'RESET' and 'AGGREGATE' could be suported here too (reset by holding a serialized initial opt state)
        self.opt_treatment = 'CONTINUE'

        self.checkpoint_out_path = checkpoint_out_path
        self.checkopint_in_path = checkpoint_in_path
        # TODO: Figure out model initialization (compute out of band and distribute to latest model path? Best if put on a cpu before loading)
        # self.model_init_path = os.path.join(model_dir, model_init_fpath)

        self.device = device
        self.config_path = config_path

        
        
        #TODO:  Do we need to call dummy train task to initialize the optimizer?
             
        #         self.dummy_train()
        
        # TODO: Rather than load initial model, place the initial model in last model spot so it gets picked up for training (i.e. always use 'continue' when training) 

        # TODO: We'll see, may not use
        self.metrics_information = None

        self.required_tensorkeys_for_function = {}
        self.initialize_tensorkeys_for_functions()

         # TODO: Overwrite methods below (get and save tensor dict rebuild model, etc.)
     

    # defining some class methods using some util functions imported above

    def rebuild_model(self, **kwargs):
        rebuild_model_util(runner_class=self, **kwargs)

    def initialize_tensorkeys_for_functions(self, **kwargs):
        initialize_tensorkeys_for_functions_util(runner_class=self, **kwargs)
     
    def reset_opt_vars(self):
        # TODO: Idea would be to save some initial optimizer state in a dictionary that can then be used to reset with
        raise NotImplementedError()

    def set_tensor_dict(self, tensor_dict, with_opt_vars=False):
        """Set the tensor dictionary.

        Args:
            tensor_dict: The tensor dictionary
            with_opt_vars (bool): Return the tensor dictionary including the
                                  optimizer tensors (Default=False)
        """
        self.write_tensors_into_checkpoint(tensor_dict=tensor_dict, with_opt_vars=with_opt_vars, checkpoint_path=self.checkpoint_path)

    def write_tensors_into_checkpoint(self, tensor_dict, with_opt_vars, checkpoint_path):
        """
        Save model state in tensor_dict to in a pickle file at checkpoint_path. Uses pt.save(). 
        All state in the checkpoint other than the model state will be kept as is in the file.
        Note: Utilization of a with_opt_vars input will be needed (along with saving an initial state optimizer state on disk),
              will be needed if a self.opt_treatement of 'RESET' or 'AGG' are to be used 
        
            Here is an example of a dictionary NNUnet uses for its state:
            save_this = 
                {
                'epoch': self.epoch + 1,
                'state_dict': state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'lr_scheduler_state_dict': lr_sched_state_dct,
                'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                           self.all_val_eval_metrics),
                'best_stuff' : (self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA)
                }


        Args:
            tensor_dict (dictionary)                 : Dictionary with keys 
            checkpoint_path (string)                 : Path to pickle file to be
                                                       created by pt.save().
            copy_path (string)                : path to checkpoint file used to populate key value pairs other than for keys: self.model_state_dict_key
                                                and self.optimizer_state_dict_key
            kwargs                            : unused

        Returns:
            epoch
        """
        # TODO: For now leaving the lr_scheduler_state_dict unchanged
        # get device for correct placement of tensors
        device = self.device

        checkpoint_dict = torch.load(checkpoint_path, map_location=device)
        epoch = checkpoint_dict['epoch']
        new_state = {}
        # grabbing keys from tensor_dict helps to double check we are getting all of those keys
        for k in tensor_dict:
            new_state[k] = torch.from_numpy(tensor_dict[k]).to(device)
        checkpoint_dict['state_dict'] = new_state
        
        # TODO: We should test this for 'RESET', 'CONTINUE', and 'AGG'
        if with_opt_vars:
            # see if there is state to restore first
            if tensor_dict.pop('__opt_state_needed') == 'true':
                checkpoint_dict = self._set_optimizer_state(derived_opt_state_dict=tensor_dict, 
                                                            checkpoint_dict=checkpoint_dict)
        torch.save(checkpoint_dict, checkpoint_path)
        # we may want to know epoch so that we can properly tell the training script to what epoch to train (NNUnet V1 only supports training with a max_num_epochs setting)
        return epoch



    def get_tensor_dict(self, with_opt_vars=False):
        """Return the tensor dictionary.

        Args:
            with_opt_vars (bool): Return the tensor dictionary including the
                                optimizer tensors (Default=False)

        Returns:
            dict: Tensor dictionary {**dict, **optimizer_dict}

        """
        self.read_tensors_from_checkpoint(with_opt_vars=with_opt_vars, checkpoint_path=self.checkpoint_path)

    def read_tensors_from_checkpoint(with_opt_vars, checkpoint_path):
        """Return a tensor dictionary interpreted from a checkpoint.

        Args:
            with_opt_vars (bool): Return the tensor dictionary including the
                                optimizer tensors (Default=False)

        Returns:
            dict: Tensor dictionary {**dict, **optimizer_dict}

        """
        pickle_dict = torch.load(checkpoint_path)

        state = to_cpu_numpy(pickle_dict['state_dict'])

        if with_opt_vars:
            raise NotImplementedError(f"Need to support putting optimizer state from aggregator for example into class attribute")
            opt_state = self.optimizer_state
            state = {**state, **opt_state}

        return state



    def _get_weights_names(self, with_opt_vars=False):
        """
        Gets model and potentially optimizer state dict key names
        args:
        with_opt_vars(bool) : Wether or not to get the optimizer key names 
        """
        state = self.get_tensor_dict(with_opt_vars=with_opt_vars)
        return state.keys()

    def _set_optimizer_state(self, derived_opt_state_dict, checkpoint_dict):
        """Set the optimizer state.
        # TODO: Refactor this, we will sparate the custom aspect of the checkpoint dict from the more general code 

        Args:
            derived_opt_state_dict(bool)    : flattened optimizer state dict
            checkpoint_dict(dict)           : checkpoint dictionary

        """
        temp_state_dict = expand_derived_opt_state_dict(derived_opt_state_dict, device=self.device)
        # Note: The expansion above only populates the 'params' key of each param group under opt_state_dict['param_groups']
        #       Therefore the default values under the additional keys such as: 'lr', 'momentum', 'dampening', 'weight_decay', 'nesterov', 'maximize', 'foreach', 'differentiable'
        #       need to be held over from the their initial values.
        # FIXME: Figure out whether or not this breaks learning rate scheduling and the like.

        # Letting default values (everything under optimizer.state_dict['param_groups']) stay unchanged (these
        # are not contained in the temp_state_dict
        # Assuming therefore that the optimizer.defaults are not changing over course of training. 
        # Therefore 'param_groups' key is remained unchanged except for 'params' key, value under it and 
        # we only modify the 'state' key value pairs otherwise
        for group_idx, group in enumerate(temp_state_dict['param_groups']):
            checkpoint_dict['optimizer_state_dict']['params'] = derived_opt_state_dict['params']
        checkpoint_dict['optimizer_state_dict']['state'] = derived_opt_state_dict['state']

        return checkpoint_dict

    def _get_optimizer_state():


        
    def train(self, col_name, round_num, input_tensor_dict, epochs=2, **kwargs):
        """Perform training for a specified number of epochs."""

        # will have below, but for now implementing this to test a child class instance
        # raise NotImplementedError()

       

        #         if 'metrics' not in kwargs:
        #             raise KeyError('metrics must be included in kwargs')
        #         param_metrics = kwargs['metrics']

        self.rebuild_model(round_num, input_tensor_dict)
        # TODO: Is it ok that I am not giving aggregated metrics to the checkpoint file?
        # 1. Insert tensor_dict info into checkpoint
        epoch = self.set_tensor_dict(tensor_dict=input_tensor_dict, with_opt_vars=False)
        # 2. Train function existing externally
        # Some todo inside function below
        # TODO: test for off-by-one error
        # TODO: we need to disable validation if possible, and separately call validation  
        train_nnunet(epochs=epochs, current_epoch=epoch)
       
        """
        # This is actual code to use later that calls an external procedure
        if 'train_fun' in kwargs:
            train_function = get_train_function('train_fun')
            train_function(data_path="", round_num=round_num, epochs=epochs) # config_path, model_dir, model_name, device='cuda:0', epochs=2
        else:
            # TODO: Will fill this in later
            
            proc = subprocess.run(["mlcube_docker",
                                "run",
                                "--mlcube={}".format(self.mlcube_dir),
                                "--platform={}".format(platform_yaml),
                                "--task={}".format(task_yaml)])
        """
        
        # 3. Load model from native format
        
        metrics = self.load_native(self.round_to_model_endpath(model_dir=self.model_dir, model_name=self.model_name, epochs=epochs, round=round_num))

        # set the training data size
        sample_count = int(metrics.pop(self.training_sample_count_key))
        self.data_loader.set_train_data_size(sample_count)

        # 5. Convert to tensorkeys

        # output metric tensors (scalar)
        origin = col_name
        tags = ('trained',)
        output_metric_dict = {
            TensorKey(
                metric_name, origin, round_num, True, ('metric',)
            ): np.array(
                    metrics[metric_name]
                ) for metric_name in metrics}

        # output model tensors (Doesn't include TensorKey)
        output_model_dict = self.get_tensor_dict(with_opt_vars=True)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            self.logger, output_model_dict,
            **self.tensor_dict_split_fn_kwargs
        )

        # create global tensorkeys
        global_tensorkey_model_dict = {
            TensorKey(
                tensor_name, origin, round_num, False, tags
            ): nparray for tensor_name, nparray in global_model_dict.items()
        }
        # create tensorkeys that should stay local
        local_tensorkey_model_dict = {
            TensorKey(
                tensor_name, origin, round_num, False, tags
            ): nparray for tensor_name, nparray in local_model_dict.items()
        }
        # the train/validate aggregated function of the next round will look
        # for the updated model parameters.
        # this ensures they will be resolved locally
        next_local_tensorkey_model_dict = {
            TensorKey(
                tensor_name, origin, round_num + 1, False, ('model',)
            ): nparray for tensor_name, nparray in local_model_dict.items()
        }

        global_tensor_dict = {
            **output_metric_dict,
            **global_tensorkey_model_dict
        }
        local_tensor_dict = {
            **local_tensorkey_model_dict,
            **next_local_tensorkey_model_dict
        }

        # update the required tensors if they need to be pulled from the
        # aggregator
        # TODO this logic can break if different collaborators have different
        #  roles between rounds.
        # for example, if a collaborator only performs validation in the first
        # round but training in the second, it has no way of knowing the
        # optimizer state tensor names to request from the aggregator
        # because these are only created after training occurs.
        # A work around could involve doing a single epoch of training
        # on random data to get the optimizer names, and then throwing away
        # the model.
        if self.opt_treatment == 'CONTINUE_GLOBAL':
            self.initialize_tensorkeys_for_functions(with_opt_vars=True)

        return global_tensor_dict, local_tensor_dict

        

    def validate(self, col_name, round_num, input_tensor_dict, **kwargs):
        """
        Run the trained model on validation data; report results.

        Parameters
        ----------
        input_tensor_dict : either the last aggregated or locally trained model

        Returns
        -------
        output_tensor_dict : {TensorKey: nparray} (these correspond to acc,
         precision, f1_score, etc.)
        """

        raise NotImplementedError()

        """ - TBD - for now commenting out

        self.rebuild_model(round_num, input_tensor_dict, validation=True)

        # 1. Save model in native format
        self.save_native(self.mlcube_model_in_path)

        # 2. Call MLCube validate task
        platform_yaml = os.path.join(self.mlcube_dir, 'platforms', '{}.yaml'.format(self.mlcube_runner_type))
        task_yaml = os.path.join(self.mlcube_dir, 'run', 'evaluate.yaml')
        proc = subprocess.run(["mlcube_docker",
                               "run",
                               "--mlcube={}".format(self.mlcube_dir),
                               "--platform={}".format(platform_yaml),
                               "--task={}".format(task_yaml)])

        # 3. Load any metrics
        metrics = self.load_metrics(os.path.join(self.mlcube_dir, 'workspace', 'metrics', 'evaluate_metrics.json'))

        # set the validation data size
        sample_count = int(metrics.pop(self.evaluation_sample_count_key))
        self.data_loader.set_valid_data_size(sample_count)

        # 4. Convert to tensorkeys
    
        origin = col_name
        suffix = 'validate'
        if kwargs['apply'] == 'local':
            suffix += '_local'
        else:
            suffix += '_agg'
        tags = ('metric', suffix)
        output_tensor_dict = {
            TensorKey(
                metric_name, origin, round_num, True, tags
            ): np.array(metrics[metric_name])
            for metric_name in metrics
        }

        return output_tensor_dict, {}

        """


    def load_metrics(self, filepath):
        """
        Load metrics from file on disk
        """
        raise NotImplementedError()
        """
        with open(filepath) as json_file:
            metrics = json.load(json_file)
        return metrics
        """