import argparse
import os

from fl_setup import main as setup_fl

def list_of_strings(arg):
    return arg.split(',')

def get_task_folder_names(first_three_digit_task_num, num_institutions, task_name, overwrite_nnunet_datadirs):
    """
    Creates task folders for all simulated instiutions in the federation
    """
    nnunet_dst_pardirs = []
    nnunet_images_train_pardirs = []
    nnunet_labels_train_pardirs = []

    task_nums = range(first_three_digit_task_num, first_three_digit_task_num + num_institutions)
    tasks = [f'Task{str(num)}_{task_name}' for num in task_nums]
    for task in tasks:

        # The NNUnet data path is obtained from an environmental variable
        nnunet_dst_pardir = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', f'{task}')
            
        nnunet_images_train_pardir = os.path.join(nnunet_dst_pardir, 'imagesTr')
        nnunet_labels_train_pardir = os.path.join(nnunet_dst_pardir, 'labelsTr')

        if os.path.exists(nnunet_images_train_pardir) and os.path.exists(nnunet_labels_train_pardir):
            raise ValueError(f"Train images pardirs: {nnunet_images_train_pardir} and {nnunet_labels_train_pardir} both already exist. Please move them both and rerun to prevent overwriting.")
        elif os.path.exists(nnunet_images_train_pardir):
            raise ValueError(f"Train images pardir: {nnunet_images_train_pardir} already exists, please move and run again to prevent overwriting.")
        elif os.path.exists(nnunet_labels_train_pardir):
            raise ValueError(f"Train labels pardir: {nnunet_labels_train_pardir} already exists, please move and run again to prevent overwriting.") 
        
        nnunet_dst_pardirs.append(nnunet_dst_pardir)
        nnunet_images_train_pardirs.append(nnunet_images_train_pardir)
        nnunet_labels_train_pardirs.append(nnunet_labels_train_pardir)

    return task_nums, tasks, nnunet_dst_pardirs, nnunet_images_train_pardirs, nnunet_labels_train_pardirs

def main(postopp_pardirs, 
         first_three_digit_task_num, 
         init_model_path, 
         init_model_info_path,
         plans_path, 
         task_name, 
         percent_train, 
         split_logic, 
         network, 
         network_trainer, 
         fold, 
         timestamp_selection='latest', 
         num_institutions=1, 
         cuda_device='0',
         overwrite_nnunet_datadirs=False,
         verbose=False):
    """
    Generates symlinks to be used for NNUnet training, assuming we already have a 
    dataset on file coming from MLCommons RANO experiment data prep.

    Also creates the json file for the data, as well as runs nnunet preprocessing.

    should be run using a virtual environment that has nnunet version 1 installed.

    args:
    postopp_pardirs(list of str)     : Parent directories for postopp data. The length of the list should either be 
                                   equal to num_insitutions, or one. If the length of the list is one and num_insitutions is not one,
                                   the samples within that single directory will be used to create num_insititutions shards.
                                   If the length of this list is equal to num_insitutions, the shards are defined by the samples within each string path.  
                                   Either way, all string paths within this list should piont to folders that have 'data' and 'labels' subdirectories with structure:
                                    ├── data
                                    │   ├── AAAC_0
                                    │   │   ├── 2008.03.30
                                    │   │   │   ├── AAAC_0_2008.03.30_brain_t1c.nii.gz
                                    │   │   │   ├── AAAC_0_2008.03.30_brain_t1n.nii.gz
                                    │   │   │   ├── AAAC_0_2008.03.30_brain_t2f.nii.gz
                                    │   │   │   └── AAAC_0_2008.03.30_brain_t2w.nii.gz
                                    │   │   └── 2008.12.17
                                    │   │       ├── AAAC_0_2008.12.17_brain_t1c.nii.gz
                                    │   │       ├── AAAC_0_2008.12.17_brain_t1n.nii.gz
                                    │   │       ├── AAAC_0_2008.12.17_brain_t2f.nii.gz
                                    │   │       └── AAAC_0_2008.12.17_brain_t2w.nii.gz
                                    │   ├── AAAC_1
                                    │   │   ├── 2008.03.30_duplicate
                                    │   │   │   ├── AAAC_1_2008.03.30_duplicate_brain_t1c.nii.gz
                                    │   │   │   ├── AAAC_1_2008.03.30_duplicate_brain_t1n.nii.gz
                                    │   │   │   ├── AAAC_1_2008.03.30_duplicate_brain_t2f.nii.gz
                                    │   │   │   └── AAAC_1_2008.03.30_duplicate_brain_t2w.nii.gz
                                    │   │   └── 2008.12.17_duplicate
                                    │   │       ├── AAAC_1_2008.12.17_duplicate_brain_t1c.nii.gz
                                    │   │       ├── AAAC_1_2008.12.17_duplicate_brain_t1n.nii.gz
                                    │   │       ├── AAAC_1_2008.12.17_duplicate_brain_t2f.nii.gz
                                    │   │       └── AAAC_1_2008.12.17_duplicate_brain_t2w.nii.gz
                                    │   ├── AAAC_extra
                                    │   │   └── 2008.12.10
                                    │   │       ├── AAAC_extra_2008.12.10_brain_t1c.nii.gz
                                    │   │       ├── AAAC_extra_2008.12.10_brain_t1n.nii.gz
                                    │   │       ├── AAAC_extra_2008.12.10_brain_t2f.nii.gz
                                    │   │       └── AAAC_extra_2008.12.10_brain_t2w.nii.gz
                                    │   ├── data.csv
                                    │   └── splits.csv
                                    ├── labels
                                    │   ├── AAAC_0
                                    │   │   ├── 2008.03.30
                                    │   │   │   └── AAAC_0_2008.03.30_final_seg.nii.gz
                                    │   │   └── 2008.12.17
                                    │   │       └── AAAC_0_2008.12.17_final_seg.nii.gz
                                    │   ├── AAAC_1
                                    │   │   ├── 2008.03.30_duplicate
                                    │   │   │   └── AAAC_1_2008.03.30_duplicate_final_seg.nii.gz
                                    │   │   └── 2008.12.17_duplicate
                                    │   │       └── AAAC_1_2008.12.17_duplicate_final_seg.nii.gz
                                    │   └── AAAC_extra
                                    │       └── 2008.12.10
                                    │           └── AAAC_extra_2008.12.10_final_seg.nii.gz
                                    └── report.yaml

    first_three_digit_task_num(str) : Should start with '5'. If fedsim == N, all N task numbers starting with this number will be used.
    init_model_path (str)           : path to initial (pretrained) model file [default None] - must be provided if init_model_info_path is.
                                      [ONLY USE IF YOU KNOW THE MODEL ARCHITECTURE MAKES SENSE FOR THE FEDERATION. OTHERWISE ARCHITECTURE IS CHOSEN USING COLLABORATOR 0 DATA.]
    init_model_info_path(str)       : path to initial (pretrained) model info pikle file [default None]- must be provided if init_model_path is.
                                      [ONLY USE IF YOU KNOW THE MODEL ARCHITECTURE MAKES SENSE FOR THE FEDERATION. OTHERWISE ARCHITECTURE IS CHOSEN USING COLLABORATOR 0 DATA.]
    plans_path(str)                 : path to initial (pretrained) model plans file [default None]- must be provided if init_model_path is.
                                      [ONLY USE IF YOU KNOW THE MODEL ARCHITECTURE MAKES SENSE FOR THE FEDERATION. OTHERWISE ARCHITECTURE IS CHOSEN USING COLLABORATOR 0 DATA.]
    task_name(str)                  : Name of task that is part of the task name
    percent_train(float)            : The percentage of samples to split into the train portion for the fold specified below (NNUnet makes its own folds but we overwrite
                                      all with None except the fold indicated below and put in our own split instead determined by a hard coded split logic default)
    split_logic(str)                : Determines how the percent_train is computed. Choices are: 'by_subject' and 'by_subject_time_pair' (see inner function docstring)
    network(str)                    : NNUnet network to be used
    network_trainer(str)            : NNUnet network trainer to be used
    fold(str)                       : Fold to train on, can be a sting indicating an int, or can be 'all'
    task_name(str)                  : Any string task name.
    timestamp_selection(str)        : Indicates how to determine the timestamp to pick. Only 'earliest', 'latest', or 'all' are supported.
                                      for each subject ID at the source: 'latest' and 'earliest' are the only ones supported so far
    num_institutions(int)           : Number of simulated institutions to shard the data into.
    verbose(bool)                   : If True, print debugging information.
    overwrite_nnunet_datadirs(bool) : Allows for overwriting past instances of NNUnet data directories using the task numbers from first_three_digit_task_num to that plus one less than number of insitutions
    """
    
    # some argument inspection
    task_digit_length = len(str(first_three_digit_task_num))
    if task_digit_length != 3:
         raise ValueError(f'The number of digits in {first_three_digit_task_num} should be 3, but it is: {task_digit_length} instead.')
    if str(first_three_digit_task_num)[0] != '5':
         raise ValueError(f"The three digit task number: {first_three_digit_task_num} should start with 5 to avoid NNUnet repository tasks, but it starts with {first_three_digit_task_num[0]}")    
    if init_model_path or init_model_info_path:
          if not init_model_path or not init_model_info_path:
                raise ValueError(f"If either init_model_path or init_model_info_path are provided, they both must be.")
    if init_model_path:
          if not init_model_path.endswith('.model'):
                raise ValueError(f"Initial model file should end with, '.model'")
          if not init_model_info_path.endswith('.model.pkl'):
                raise ValueError(f"Initial model info file should end with, 'model.pkl'")
          
    task_nums = range(first_three_digit_task_num, first_three_digit_task_num + num_institutions)

    # task_folder_info is a zipped lists indexed over tasks (collaborators)
    for col_idx, (task_num, postopp_pardir) in enumerate(zip(task_nums,postopp_pardirs)):
        print(f"\n\n##############\n\nSettup up for postopp_pardir: {postopp_pardir}\n\n##################\n\n")
        if col_idx == 0:
            col_paths = setup_fl(postopp_pardir=postopp_pardir, 
                                  three_digit_task_num=task_num,  
                                  task_name=task_name, 
                                  percent_train=percent_train, 
                                  split_logic=split_logic, 
                                  network=network, 
                                  network_trainer=network_trainer, 
                                  fold=fold,
                                  init_model_path=init_model_path, 
                                  init_model_info_path=init_model_info_path,
                                  plans_path=plans_path, 
                                  timestamp_selection=timestamp_selection, 
                                  cuda_device=cuda_device,
                                  overwrite_nnunet_datadirs=overwrite_nnunet_datadirs, 
                                  verbose=verbose)
        else: 
            if not init_model_path:
                if init_model_info_path or plans_path:
                    raise ValueError(f"If init_model_path is not provided, then init_model_info_path and plans_path are also not expected.")
                init_model_path = col_paths['initial_model_path']
                init_model_info_path = col_paths['initial_model_info_path']
                plans_path = col_paths['plans_path']

            setup_fl(postopp_pardir=postopp_pardir, 
                        three_digit_task_num=task_num,  
                        task_name=task_name, 
                        percent_train=percent_train, 
                        split_logic=split_logic, 
                        network=network, 
                        network_trainer=network_trainer, 
                        fold=fold,
                        init_model_path=init_model_path, 
                        init_model_info_path=init_model_info_path,
                        plans_path=plans_path, 
                        timestamp_selection=timestamp_selection, 
                        cuda_device=cuda_device,
                        overwrite_nnunet_datadirs=overwrite_nnunet_datadirs, 
                        verbose=verbose)
        


if __name__ == '__main__':

        argparser = argparse.ArgumentParser()
        argparser.add_argument(
            '--postopp_pardirs',
            type=list_of_strings,
            # nargs='+',
            help="Parent directories to postopp data (all should have 'data' and 'labels' subdirectories). Length needs to equal num_institutions or be lengh 1.")
        argparser.add_argument(
            '--first_three_digit_task_num',
            type=int,
            help="Should start with '5'. If fedsim == N, all N task numbers starting with this number will be used.")
        argparser.add_argument(
            '--init_model_path',
            type=str,
            default=None,
            help="Path to initial (pretrained) model file [ONLY USE IF YOU KNOW THE MODEL ARCHITECTURE MAKES SENSE FOR THE FEDERATION. OTHERWISE ARCHITECTURE IS CHOSEN USING COLLABORATOR 0's DATA.].")
        argparser.add_argument(
            '--init_model_info_path',
            type=str,
            default=None,
            help="Path to initial (pretrained) model info file [ONLY USE IF YOU KNOW THE MODEL ARCHITECTURE MAKES SENSE FOR THE FEDERATION. OTHERWISE ARCHITECTURE IS CHOSEN USING COLLABORATOR 0's DATA.].")
        argparser.add_argument(
            '--plans_path',
            type=str,
            default=None,
            help="Path to the plans file [ONLY USE IF YOU KNOW THE MODEL ARCHITECTURE MAKES SENSE FOR THE FEDERATION. OTHERWISE ARCHITECTURE IS CHOSEN USING COLLABORATOR 0's DATA.].")
        argparser.add_argument(
            '--task_name',
            type=str,
            help="Part of the NNUnet data task directory name. With 'first_three_digit_task_num being 'XXX', the directory name becomes: .../nnUNet_raw_data_base/nnUNet_raw_data/TaskXXX_<task_name>.")
        argparser.add_argument(
            '--percent_train',
            type=float,
            default=0.8,
            help="The percentage of samples to split into the train portion for the fold specified below (NNUnet makes its own folds but we overwrite) - see docstring in main")
        argparser.add_argument(
          '--split_logic',
            type=str,
            default='by_subject_time_pair',
            help="Determines how the percent_train is computed. Choices are: 'by_subject' and 'by_subject_time_pair' (see inner function docstring)")
        argparser.add_argument(
          '--network',
            type=str,
            default='3d_fullres',
            help="NNUnet network to be used.")
        argparser.add_argument(
            '--network_trainer',
            type=str,
            default='nnUNetTrainerV2',
            help="NNUnet network trainer to be used.")
        argparser.add_argument(
            '--fold',
            type=str,
            default='0',
            help="Fold to train on, can be a sting indicating an int, or can be 'all'.")
        argparser.add_argument(
            '--timestamp_selection',
            type=str,
            default='all',
            help="Indicates how to determine the timestamp to pick for each subject ID at the source: 'latest' and 'earliest' are the only ones supported so far.")        
        argparser.add_argument(
            '--num_institutions',
            type=int,
            default=1,
            help="Number of symulated insitutions to shard the data into.")
        argparser.add_argument(
            '--cuda_device',
            type=str,
            default='0',
            help="Used for the setting of os.environ['CUDA_VISIBLE_DEVICES']") 
        argparser.add_argument(
            '--verbose',
            action='store_true',
            help="Print debuging information.")
        argparser.add_argument(
            '--overwrite_nnunet_datadirs',
            action='store_true',
            help="Allows overwriting NNUnet directories with task numbers from first_three_digit_task_num to that plus one les than number of insitutions.")    

        args = argparser.parse_args()

        kwargs = vars(args)

        main(**kwargs)
