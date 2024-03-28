
import os
import subprocess

from nnunet.dataset_conversion.utils import generate_dataset_json


num_to_modality = {'_0000': '_brain_t1n.nii.gz',
                   '_0001': '_brain_t2w.nii.gz',
                   '_0002': '_brain_t1c.nii.gz',
                   '_0003': '_brain_t2f.nii.gz'}


def subject_time_to_mask_path(pardir, subject, timestamp):
    mask_fname = f'{subject}_{timestamp}_tumorMask_model_0.nii.gz'
    return os.path.join(pardir, 'labels', '.tumor_segmentation_backup', subject, timestamp,'TumorMasksForQC', mask_fname)


def create_task_folders(first_three_digit_task_num, num_institutions, task_name):
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
        
        os.makedirs(nnunet_images_train_pardir, exist_ok=False)
        os.makedirs(nnunet_labels_train_pardir, exist_ok=False) 
        
        nnunet_dst_pardirs.append(nnunet_dst_pardir)
        nnunet_images_train_pardirs.append(nnunet_images_train_pardir)
        nnunet_labels_train_pardirs.append(nnunet_labels_train_pardir)

    return task_nums, tasks, nnunet_dst_pardirs, nnunet_images_train_pardirs, nnunet_labels_train_pardirs
    

def symlink_one_subject(postopp_subject_dir, postopp_data_dirpath, postopp_labels_dirpath, nnunet_images_train_pardir, nnunet_labels_train_pardir, timestamp_selection):
    postopp_subject_dirpath = os.path.join(postopp_data_dirpath, postopp_subject_dir)
    timestamps = sorted(list(os.listdir(postopp_subject_dirpath)))
    if timestamp_selection == 'latest':
        timestamp = timestamps[-1]
    elif timestamp_selection == 'earliest':
        timestamp = timestamps[0]
    else:
        raise ValueError(f"timestamp_selection currently only supports 'latest' and 'earliest', but you have requested: '{timestamp_selection}'")
            
    postopp_subject_timestamp_dirpath = os.path.join(postopp_subject_dirpath, timestamp)
    postopp_subject_timestamp_label_dirpath = os.path.join(postopp_labels_dirpath, postopp_subject_dir, timestamp)
    if not os.path.exists(postopp_subject_timestamp_label_dirpath):
        raise ValueError(f"Subject label file for data at: {postopp_subject_timestamp_dirpath} was not found in the expected location: {postopp_subject_timestamp_label_dirpath}")
    
    timed_subject = postopp_subject_dir + '_' + timestamp

    # Copy label first
    label_src_path = os.path.join(postopp_subject_timestamp_label_dirpath, timed_subject + '_final_seg.nii.gz')
    label_dst_path = os.path.join(nnunet_labels_train_pardir, timed_subject + '.nii.gz')
    os.symlink(src=label_src_path, dst=label_dst_path)

    # Copy images
    for num in num_to_modality:
        src_path = os.path.join(postopp_subject_timestamp_dirpath, timed_subject + num_to_modality[num])
        dst_path = os.path.join(nnunet_images_train_pardir,timed_subject + num + '.nii.gz')
        os.symlink(src=src_path, dst=dst_path)


def setup_fedsim_data(postopp_pardir, first_three_digit_task_num, task_name, timestamp_selection='latest', num_institutions=1):
    """
    Generates symlinks to be used for NNUnet training, assuming we already have a 
    dataset on file coming from MLCommons RANO experiment data prep.

    Also creates the json file for the data, as well as runs nnunet preprocessing.

    should be run using a virtual environment that has nnunet version 1 installed.

    args:
    postopp_src_pardir(str)     : Parent directory for postopp data.  
                                    Should have 'data' and 'labels' subdirectories with structure:
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

    first_three_digit_task_num(str): Should start with '5'. If num_institutions == N (see below), all N task numbers starting with this number will be used.
    task_name(str)                 : Any string task name.
    timestamps(str)                : Indicates how to determine the timestamp to pick
                                   for each subject ID at the source: 'latest' and 'earliest' are the only ones supported so far
    num_institutions(int)          : Number of simulated institutions to shard the data into.

    Returns:
    task_nums, tasks, nnunet_dst_pardirs, nnunet_images_train_pardirs, nnunet_labels_train_pardirs 
    """

    task_nums, tasks, nnunet_dst_pardirs, nnunet_images_train_pardirs, nnunet_labels_train_pardirs = \
        create_task_folders(first_three_digit_task_num=first_three_digit_task_num, 
                            num_institutions=num_institutions, 
                            task_name=task_name)
    
    postopp_subdirs = list(os.listdir(postopp_pardir))
    if 'data' not in postopp_subdirs:
         raise ValueError(f"'data' must be a subdirectory of postopp_src_pardir:{postopp_pardir}, but it is not.")
    if 'labels' not in postopp_subdirs:
         raise ValueError(f"'labels' must be a subdirectory of postopp_src_pardir:{postopp_pardir}, but it is not.")

    postopp_data_dirpath = os.path.join(postopp_pardir, 'data')
    postopp_labels_dirpath = os.path.join(postopp_pardir, 'labels')

    all_subjects = list(os.listdir(postopp_data_dirpath))
    subject_shards = [all_subjects[start::num_institutions] for start in range(num_institutions)]
    
    for shard_idx, (postopp_subject_dirs, task_num, task, nnunet_dst_pardir, nnunet_images_train_pardir, nnunet_labels_train_pardir) in enumerate(zip(subject_shards, task_nums, tasks, nnunet_dst_pardirs, nnunet_images_train_pardirs, nnunet_labels_train_pardirs)):
        print(f"\n######### CREATING SYMLINKS TO POSTOPP DATA FOR COLLABORATOR {shard_idx} #########\n") 
        for postopp_subject_dir in postopp_subject_dirs:
            symlink_one_subject(postopp_subject_dir=postopp_subject_dir, 
                                postopp_data_dirpath=postopp_data_dirpath, 
                                postopp_labels_dirpath=postopp_labels_dirpath, 
                                nnunet_images_train_pardir=nnunet_images_train_pardir, 
                                nnunet_labels_train_pardir=nnunet_labels_train_pardir, 
                                timestamp_selection=timestamp_selection)
            
        # Generate json file for the dataset
        print(f"\n######### GENERATING DATA JSON FILE FOR COLLABORATOR {shard_idx} #########\n")
        json_path = os.path.join(nnunet_dst_pardir, 'dataset.json')
        labels = {0: 'Background', 1: 'Necrosis', 2: 'Edema', 3: 'Enhancing Tumor', 4: 'Cavity'}
        # labels = {0: 'Background', 1: 'Necrosis', 2: 'Edema'}
        generate_dataset_json(output_file=json_path, imagesTr_dir=nnunet_images_train_pardir, imagesTs_dir=None, modalities=tuple(num_to_modality.keys()),
                            labels=labels, dataset_name='RANO Postopp')
        
        # Now call the os process to preprocess the data
        print(f"\n######### OS CALL TO PREPROCESS DATA FOR COLLABORATOR {shard_idx} #########\n")
        subprocess.run(["nnUNet_plan_and_preprocess",  "-t",  f"{task_num}", "--verify_dataset_integrity"])

    return tasks