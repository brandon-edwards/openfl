# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Contributors: Micah Sheller, Brandon Edwards  - DELETEME?

"""

"""You may copy this file as the starting point of your own model."""

import json
import os

class NNUNetDummyDataLoader():
    def __init__(self, data_path, p_train, partial_epoch):
        self.task_name = data_path
        data_base_path = os.path.join(os.environ['nnUNet_preprocessed'], self.task_name)
        with open(f'{data_base_path}/dataset.json', 'r') as f:
            data_config = json.load(f)
        data_size = data_config['numTraining']

        # NOTE: The data_size above is the exact number of brain volumes before the train/val split (using p_train).
        #       The train_data_size and valid_data_size attributes (below) are the weightings applied to this collaborator in aggregation
        #       of training and validation results.
        #       NNUnet train and val loaders sample batches with replacement! The task runner we use specifies
        #       the batches counts self.train_data_size/batch_size and self.valid_data_size/batch_size respectively
        #       When partial_epoch is 1.0, this makes the expected number of times a training or validation point
        #       to be used during training to be equal to the number of epochs(rounds).
        # TODO: determine how nnunet validation splits round
        # TODO: p_train here is decoupled with a hard coded ''
        self.train_data_size = int(partial_epoch * p_train * data_size)
        self.valid_data_size = int(partial_epoch * (1 - p_train) * data_size )

    def get_feature_shape(self):
        return [1,1,1]
    
    def get_train_data_size(self):
        return self.train_data_size
    
    def get_valid_data_size(self):
        return self.valid_data_size
    
    def get_task_name(self):
        return self.task_name
