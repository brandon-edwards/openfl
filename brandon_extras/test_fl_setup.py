import sys
import os
sys.path.append('/home/edwardsb/repositories/be-SATGOpenFL/openfl/federated/task')
from fl_setup import main as setup_fl
from fedsim_setup_using_fl_setup import main as setup_fedsim
from nnunet_v1 import train_nnunet

network = '3d_fullres'
network_trainer = 'nnUNetTrainerV2'
fold = '0'

cuda_device='7'

os.environ['CUDA_VISIBLE_DEVICES']=cuda_device

def train_on_task(task, continue_training=True, current_epoch=0, without_data_unpacking=False, use_compressed_data=False):
    print(f"###########\nStarting training for task: {task}\n")
    train_nnunet(epochs=1, 
                 current_epoch = current_epoch, 
                 network = network,
                 task=task, 
                 network_trainer = network_trainer, 
                 fold=fold, 
                 continue_training=continue_training, 
                 use_compressed_data=use_compressed_data)

  
############################################################################################################

print(f"\n##############\nRunning Test with fedsim_setup for two followed by fl_setup for one\n#################\n")
first_col_paths = setup_fedsim(postopp_pardirs=['/raid/edwardsb/projects/RANO/test_data_links_random_times_0','/raid/edwardsb/projects/RANO/test_data_links_random_times_1'],
                        first_three_digit_task_num=522,
                        task_name='MultPathTest',
                        num_institutions=2, 
                        cuda_device=cuda_device,
                        overwrite_nnunet_datadirs=True,
                        verbose=False)

setup_fl(postopp_pardir='/raid/edwardsb/projects/RANO/test_data_links_random_times_2', 
         three_digit_task_num=524,
         task_name='MultPathTest', 
         cuda_device=cuda_device, 
         plans_path=first_col_paths['plans_path'], 
         init_model_info_path=first_col_paths['final_model_info_path'], 
         init_model_path=first_col_paths['final_model_path'], 
         overwrite_nnunet_datadirs=True, 
         verbose=True)

# Now train on each collaborator
for task in ['Task522_MultPathTest', 'Task523_MultPathTest', 'Task524_MultPathTest']:
    train_on_task(task=task, current_epoch=1)


#############################################################################################################


print(f"\n##############\nRunning Test with fl_setup for one followed by fedsim_setup... for two\n#################\n")
first_col_paths = setup_fl(postopp_pardir='/raid/edwardsb/projects/RANO/test_data_links_random_times_1', 
         three_digit_task_num=522,
         task_name='MultPathTest', 
         cuda_device=cuda_device,  
         overwrite_nnunet_datadirs=True, 
         verbose=True)


setup_fedsim(postopp_pardirs=['/raid/edwardsb/projects/RANO/test_data_links_random_times_0','/raid/edwardsb/projects/RANO/test_data_links_random_times_2'],
                        first_three_digit_task_num=523,
                        task_name='MultPathTest',
                        num_institutions=2, 
                        cuda_device=cuda_device,
                        plans_path=first_col_paths['plans_path'], 
                        init_model_info_path=first_col_paths['final_model_info_path'], 
                        init_model_path=first_col_paths['final_model_path'],
                        overwrite_nnunet_datadirs=True,
                        verbose=False)

# Now train on each collaborator
for task in ['Task522_MultPathTest', 'Task523_MultPathTest', 'Task524_MultPathTest']:
    train_on_task(task=task, current_epoch=1)



################################################################################################################



print(f"\n##############\nRunning Test with thre times fl_setup for three\n#################\n")
first_col_paths = setup_fl(postopp_pardir='/raid/edwardsb/projects/RANO/test_data_links_random_times_2', 
         three_digit_task_num=522,
         task_name='MultPathTest', 
         cuda_device=cuda_device,  
         overwrite_nnunet_datadirs=True, 
         verbose=True)


setup_fl(postopp_pardir='/raid/edwardsb/projects/RANO/test_data_links_random_times_1',
                        three_digit_task_num=523,
                        task_name='MultPathTest',
                        cuda_device=cuda_device,
                        plans_path=first_col_paths['plans_path'], 
                        init_model_info_path=first_col_paths['final_model_info_path'], 
                        init_model_path=first_col_paths['final_model_path'],
                        overwrite_nnunet_datadirs=True,
                        verbose=False)

setup_fl(postopp_pardir='/raid/edwardsb/projects/RANO/test_data_links_random_times_0',
                        three_digit_task_num=524,
                        task_name='MultPathTest',
                        cuda_device=cuda_device,
                        plans_path=first_col_paths['plans_path'], 
                        init_model_info_path=first_col_paths['final_model_info_path'], 
                        init_model_path=first_col_paths['final_model_path'],
                        overwrite_nnunet_datadirs=True,
                        verbose=False)


# Now train on each collaborator
for task in ['Task522_MultPathTest', 'Task523_MultPathTest', 'Task524_MultPathTest']:
    train_on_task(task=task, current_epoch=1)



print("##############################################################################################\n")
print("##############################################################################################\n")
print("                            ALL TESTS PASSED\n")
print("##############################################################################################\n")
print("##############################################################################################\n")