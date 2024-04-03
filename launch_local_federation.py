# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import socket
import argparse
import yaml
from pathlib import Path
from subprocess import check_call
from concurrent.futures import ProcessPoolExecutor

from openfl.utilities.utils import rmtree
from tests.github.utils import create_collaborator, create_certified_workspace, certify_aggregator


def main(gpu_base=0, fed_yaml='local_federation.yaml', **kwargs):
    with open(fed_yaml, 'r') as f:
        config = yaml.safe_load(f)
    print("running federation with config:")
    for k, v in config.items():
        print(f'{k}:{v}')
    
    origin_dir = Path.cwd().resolve()
    fed_workspace = config['fed_workspace']
    archive_name = f'{fed_workspace}.zip'
    fqdn = socket.getfqdn()
    template = config['template']
    rounds_to_train = config['rounds_to_train']
    cols = config['cols']

    # START
    # =====
    # Make sure you are in a Python virtual environment with the FL package installed.
    create_certified_workspace(fed_workspace, template, fqdn, rounds_to_train)
    certify_aggregator(fqdn)

    workspace_root = Path().resolve()  # Get the absolute directory path for the workspace

    for col, data_path in cols.items():
        # Create collaborator
        create_collaborator(col, workspace_root, data_path, archive_name, fed_workspace)

    # Run the federation
    with ProcessPoolExecutor(max_workers=(len(cols.keys())+1)) as executor:
        executor.submit(check_call, ['fx', 'aggregator', 'start'], cwd=workspace_root)
        time.sleep(5)

        for i, col in enumerate(cols.keys()):
            dir = workspace_root / col / fed_workspace
            col_config = {
                'gpu_num_string':i + gpu_base,
                'nnunet_task':cols[col]
            }
            with open(os.path.join(dir, 'nnunet_collaborator_config.yaml'), 'w') as f:
                yaml.safe_dump(col_config, f)
            executor.submit(check_call, ['fx', 'collaborator', 'start', '-n', col], cwd=dir)
            time.sleep(2)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--gpu_base',
        type=int,
        default=0,
        help="First GPU to use. Second+ collaborators go on the next GPUs")
    argparser.add_argument(
        '--fed_yaml',
        type=str,
        default='local_federation.yaml',
        help="yaml file that has the config for the federation")     

    args = argparser.parse_args()

    kwargs = vars(args)

    main(**kwargs)