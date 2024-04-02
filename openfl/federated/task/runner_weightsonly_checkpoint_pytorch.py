# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Contributors: Micah Sheller, Patrick Foley, Brandon Edwards  - DELETEME?

"""
# TODO: Clean up imports

import os
import subprocess
import shutil
import time
import pickle as pkl
from copy import deepcopy
import hashlib
import yaml

import numpy as np
import torch

from openfl.utilities import TensorKey
from openfl.utilities.split import split_tensor_dict_for_holdouts

from openfl.federated.task.runner import TaskRunner
from openfl.federated.task.runner_pt_utils import rebuild_model_util, derive_opt_state_dict, expand_derived_opt_state_dict
from openfl.federated.task.runner_pt_utils import initialize_tensorkeys_for_functions_util, to_cpu_numpy
from openfl.federated.task.nnunet_v1 import train_nnunet

def get_train_function():
    # TODO: Keep?
    return NotImplementedError()


class WeightsOnlyPyTorchCheckpointTaskRunner(TaskRunner):
    """An abstract class for PyTorch model based Tasks, where training, validation etc. are processes that
       pull model state from a PyTorch checkpoint."""

    def __init__(self,
                 nnunet_task=None,
                 checkpoint_out_path = None,
                 checkpoint_in_path = None,
                 checkpoint_init_path = None,
                 device = 'cuda',
                 gpu_num_string=None,
                 config_path=None,
                 **kwargs):
        """Initialize.

        Args:
            checkpoint_out_path(str)    : Path to the model checkpoint that will be used to start local training.
            checkpoint_in_path(str)     : Path to model checkpoint that results from performing local training.
            device(str)                 : Device ('cpu' or 'cuda') to be used for training and validation script computations.
            gpu_num_string(str)         : String indicating which gpu(s) to make available
            kwargs                      : Additional key work arguments (will be passed to rebuild_model, initialize_tensor_key_functions, TODO: <Fill this in>).
            config_path(str)            : Path to the configuration file used by the training and validation script.
            TODO: 
        """ 
        super().__init__(**kwargs)

        # TODO: Both 'CONTINUE_GLOBAL' and 'RESET' could be suported here too (currently RESET throws an exception related to a 
        # missmatch in size coming from the momentum buffer and other stuff either in the model or optimizer)
        self.opt_treatment = 'CONTINUE_LOCAL'

        # Load local config file
        if os.path.exists('nnunet_collaborator_config.yaml'):
            with open('nnunet_collaborator_config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            self.gpu_num_string = str(config['gpu_num_string'])
            self.nnunet_task = config['nnunet_task']
        else:
            self.gpu_num_string = str(gpu_num_string)
            self.nnunet_task = nnunet_task

        # FIXME: Understand what of this needs to be configurable for NNUNet
        base_model_path = os.path.join(os.environ['RESULTS_FOLDER'], 'nnUNet', '3d_fullres')
        base_model_path = os.path.join(base_model_path, self.nnunet_task, 'nnUNetTrainerV2__nnUNetPlansv2.1', 'fold_0')
        
        # THIS IS BROKEN UNTIL WE HAVE THIS WORKING WITH CONTAINERS
        if checkpoint_out_path is None:
            self.checkpoint_out_path = os.path.join(base_model_path, 'model_final_checkpoint.model')
        else:
            self.checkpoint_out_path = checkpoint_out_path
        
        if checkpoint_in_path is None:
            self.checkpoint_in_path = os.path.join(base_model_path, 'model_final_checkpoint.model')
        else:
            self.checkpoint_in_path = checkpoint_in_path
        
        if checkpoint_init_path is None:
            self.checkpoint_init_path = os.path.join(base_model_path, 'model_initial_checkpoint.model')
        else:
            self.checkpoint_init_path = checkpoint_init_path

        if device not in ['cpu', 'cuda']:
            raise ValueError("Device argument must be 'cpu' or 'cuda', but {device} was used instead.")
        self.device = device
        self.config_path = config_path

        self.training_round_completed = False

        # enable GPUs if appropriate
        if self.device == 'cuda' and not self.gpu_num_string:
            raise ValueError(f"If device is 'cuda' then gpu_num must be set rather than allowing to be the default None.")
        else:
            os.environ['CUDA_VISIBLE_DEVICES']=self.gpu_num_string

        # TODO: Currently we use a dummy loader and insert known values of training and validation data size.
      
        # TODO: We'll see, may not use
        self.metrics_information = None

        self.required_tensorkeys_for_function = {}
        self.initialize_tensorkeys_for_functions()

        # overwrite attribute to account for one optimizer param (in every
        # child model that does not overwrite get and set tensordict) that is
        # not a numpy array
        self.tensor_dict_split_fn_kwargs.update({
            'holdout_tensor_names': ['__opt_state_needed']
        })

        # Initialize model
        self.replace_checkpoint(self.checkpoint_init_path)
     

    def load_checkpoint(self, checkpoint_path=None):
        """
        Function used to load checkpoint from disk.
        """
        if not checkpoint_path:
            checkpoint_path = self.checkpoint_in_path
        checkpoint_dict = torch.load(checkpoint_path)
        return checkpoint_dict
    
    def save_checkpoint(self, checkpoint):
        """
        Function to save checkpoint to disk.
        """
        torch.save(checkpoint, self.checkpoint_out_path)

    # defining some class methods using some util functions imported above

    def rebuild_model(self, input_tensor_dict, **kwargs):
        rebuild_model_util(runner_class=self, input_tensor_dict=input_tensor_dict, **kwargs)

    def initialize_tensorkeys_for_functions(self, **kwargs):
        initialize_tensorkeys_for_functions_util(runner_class=self, **kwargs)
     
    def get_required_tensorkeys_for_function(self, func_name, **kwargs):
        """
        Get the required tensors for specified function that could be called \
        as part of a task. By default, this is just all of the layers and \
        optimizer of the model.

        Args:
            func_name

        Returns:
            list : [TensorKey]
        """
        if func_name == 'validate':
            local_model = 'apply=' + str(kwargs['apply'])
            return self.required_tensorkeys_for_function[func_name][local_model]
        else:
            return self.required_tensorkeys_for_function[func_name]

    def reset_opt_vars(self):
        current_checkpoint_dict = self.load_checkpoint()
        initial_checkpoint_dict = self.load_checkpoint(checkpoint_path=self.checkpoint_init_path)
        derived_opt_state_dict = self._get_optimizer_state(checkpoint_dict=initial_checkpoint_dict)
        self._set_optimizer_state(derived_opt_state_dict=derived_opt_state_dict, 
                                  checkpoint_dict=current_checkpoint_dict)

    def set_tensor_dict(self, tensor_dict, with_opt_vars=False):
        """Set the tensor dictionary.

        Args:
            tensor_dict: The tensor dictionary
            with_opt_vars (bool): Return the tensor dictionary including the
                                  optimizer tensors (Default=False)
        """
        return self.write_tensors_into_checkpoint(tensor_dict=tensor_dict, with_opt_vars=with_opt_vars)
    
    def replace_checkpoint(self, path_to_replacement):
        shutil.copyfile(src=path_to_replacement, dst=self.checkpoint_in_path)

    def write_tensors_into_checkpoint(self, tensor_dict, with_opt_vars):
        """
        Save model state in tensor_dict to in a pickle file at self.checkpoint_out_path. Uses pt.save(). 
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
            with_opt_vars (bool)                : Whether or not to save the optimizer state as well (this info will be part of the tensor dict in this case - i.e. tensor_dict = {**model_state, **opt_state})
            kwargs                            : unused

        Returns:
            epoch
        """
        # TODO: For now leaving the lr_scheduler_state_dict unchanged (this may be best though)
        # TODO: Do we want to test this for 'RESET', 'CONTINUE_GLOBAL'?

        # get device for correct placement of tensors
        device = self.device

        checkpoint_dict = torch.load(self.checkpoint_in_path, map_location=device)
        epoch = checkpoint_dict['epoch']
        new_state = {}
        # grabbing keys from the checkpoint state dict, poping from the tensor_dict
        # Brandon DEBUGGING
        seen_keys = []
        for k in checkpoint_dict['state_dict']:
            if k not in seen_keys:
                seen_keys.append(k)
            else:
                raise ValueError(f"\nKey {k} apears at least twice!!!!/n")
            new_state[k] = torch.from_numpy(tensor_dict[k]).to(device)
        checkpoint_dict['state_dict'] = new_state
        
        if with_opt_vars:
            # see if there is state to restore first
            if tensor_dict.pop('__opt_state_needed') == 'true':
                checkpoint_dict = self._set_optimizer_state(derived_opt_state_dict=tensor_dict, 
                                                            checkpoint_dict=checkpoint_dict)
        self.save_checkpoint(checkpoint_dict)
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
        return self.read_tensors_from_checkpoint(with_opt_vars=with_opt_vars)

    def read_tensors_from_checkpoint(self, with_opt_vars):
        """Return a tensor dictionary interpreted from a checkpoint.

        Args:
            with_opt_vars (bool): Return the tensor dictionary including the
                                optimizer tensors (Default=False)

        Returns:
            dict: Tensor dictionary {**dict, **optimizer_dict}

        """
        checkpoint_dict = self.load_checkpoint()
        state = to_cpu_numpy(checkpoint_dict['state_dict'])
        if with_opt_vars:
            opt_state = self._get_optimizer_state(checkpoint_dict=checkpoint_dict)
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
        self._write_optimizer_state_into_checkpoint(derived_opt_state_dict=derived_opt_state_dict,
                                              checkpoint_dict=checkpoint_dict, 
                                              checkpoint_path=self.checkpoint_out_path)
    
    def _write_optimizer_state_into_checkpoint(self, derived_opt_state_dict, checkpoint_dict, checkpoint_path):
        """Write the optimizer state contained within the derived_opt_state_dict into the checkpoint_dict,
           keeping some settings already contained within that checkpoint file the same, then write the resulting
           checkpoint back to the checkpoint path.
           TODO: Refactor this, we will separate the custom aspect of the checkpoint dict from the more general code 

        Args:
            derived_opt_state_dict(bool)    : flattened optimizer state dict
            checkpoint_dir(path)           : Path to the checkpoint file

        """
        temp_state_dict = expand_derived_opt_state_dict(derived_opt_state_dict, device=self.device)
        # Note: The expansion above only populates the 'params' key of each param group under opt_state_dict['param_groups']
        #       Therefore the default values under the additional keys such as: 'lr', 'momentum', 'dampening', 'weight_decay', 'nesterov', 'maximize', 'foreach', 'differentiable'
        #       need to be held over from the their initial values.
        # FIXME: Figure out whether or not this breaks learning rate scheduling and the like.

        # Letting default values (everything under temp_state_dict['param_groups'] except the 'params' key) 
        # stay unchanged (these are not contained in the temp_state_dict)
        # Assuming therefore that the optimizer.defaults (which hold this same info) are not changing over course of training. 
        # We only modify the 'state' key value pairs otherwise
        for group_idx, group in enumerate(temp_state_dict['param_groups']):
            checkpoint_dict['optimizer_state_dict']['param_groups'][group_idx]['params'] = group['params']
        checkpoint_dict['optimizer_state_dict']['state'] = temp_state_dict['state']

        torch.save(checkpoint_dict, checkpoint_path)

    def _get_optimizer_state(self, checkpoint_dict):
        """Get the optimizer state.
        Args:
            checkpoint_path(str)           : path to the checkpoint
        """
        return self._read_opt_state_from_checkpoint(checkpoint_dict)


    def _read_opt_state_from_checkpoint(self, checkpoint_dict):
        """Read the optimizer state from the checkpoint dict and put in tensor dict format.
        # TODO: Refactor this, we will sparate the custom aspect of the checkpoint dict from the more general code 
        """

        opt_state_dict = deepcopy(checkpoint_dict['optimizer_state_dict'])

        # Optimizer state might not have some parts representing frozen parameters
        # So we do not synchronize them
        param_keys_with_state = set(opt_state_dict['state'].keys())
        for group in opt_state_dict['param_groups']:
            local_param_set = set(group['params'])
            params_to_sync = local_param_set & param_keys_with_state
            group['params'] = sorted(params_to_sync)
        derived_opt_state_dict = derive_opt_state_dict(opt_state_dict)

        return derived_opt_state_dict


        
    def train(self, col_name, round_num, input_tensor_dict, epochs, **kwargs):
        # TODO: Figure out the right name to use for this method and the default assigner
        """Perform training for a specified number of epochs."""

        # will have below, but for now implementing this to test a child class instance
        # raise NotImplementedError()

       

        #         if 'metrics' not in kwargs:
        #             raise KeyError('metrics must be included in kwargs')
        #         param_metrics = kwargs['metrics']

        # # FIXME: REMOVE HORRIBLENESS BELOW ONCE WE HAVE CONTAINERS
        # base_model_path = os.path.join(os.environ['RESULTS_FOLDER'], 'nnUNet', '3d_fullres')
        # base_model_path = os.path.join(base_model_path, self.data_loader.get_task_name(), 'nnUNetTrainerV2__nnUNetPlansv2.1', 'fold_0')
        
        # # THIS IS BROKEN UNTIL WE HAVE THIS WORKING WITH CONTAINERS
        # self.checkpoint_out_path = os.path.join(base_model_path, 'model_final_checkpoint.model')
        # self.checkpoint_in_path = os.path.join(base_model_path, 'model_final_checkpoint.model')
        # self.checkpoint_init_path = os.path.join(base_model_path, 'model_initial_checkpoint.model')

        # print("****************")
        # print("set self.checkpoint_out_path to", self.checkpoint_out_path)
        # print("set self.checkpoint_in_path to", self.checkpoint_in_path)
        # print("set self.checkpoint_init_path to", self.checkpoint_init_path)
        # # FIXME: REMOVE HORRIBLENESS ABOVE ONCE WE HAVE CONTAINERS
        
        # get hashes before and after
        print("***** BEFORE HASH:", hashlib.md5(open(self.checkpoint_out_path,'rb').read()).hexdigest())
        self.rebuild_model(input_tensor_dict=input_tensor_dict, **kwargs)
        # 1. Insert tensor_dict info into checkpoint
        current_epoch = self.set_tensor_dict(tensor_dict=input_tensor_dict, with_opt_vars=False)
        print("***** AFTER HASH:", hashlib.md5(open(self.checkpoint_out_path,'rb').read()).hexdigest())
        print("***** CURRENT EPOCH FOR",self.checkpoint_out_path,"ARE",current_epoch,"*****")
        # 2. Train function existing externally
        # Some todo inside function below
        # TODO: test for off-by-one error
        # TODO: we need to disable validation if possible, and separately call validation  
        train_nnunet(epochs=epochs, current_epoch=current_epoch, task=self.data_loader.get_task_name())
       
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
        
        # 3. Load metrics from checkpoint
        (all_tr_losses, all_val_losses, all_val_losses_tr_mode, all_val_eval_metrics) = self.load_checkpoint()['plot_stuff']
        # these metrics are appended to the checkopint each epoch, so we select the most recent epoch
        metrics = {'train_loss': all_tr_losses[-1], 
                   'val_eval': all_val_eval_metrics[-1]}


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
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(logger=self.logger, 
                                                                             tensor_dict=output_model_dict,
                                                                             **self.tensor_dict_split_fn_kwargs)

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