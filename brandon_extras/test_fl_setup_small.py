import sys
import os
sys.path.append('/home/edwardsb/repositories/be-SATGOpenFL/openfl/federated/task')
from fl_setup import main as setup_fl
from fedsim_setup_using_fl_setup import main as setup_fedsim
from nnunet_v1 import train_nnunet

network = '3d_fullres'
network_trainer = 'nnUNetTrainerV2'
fold = '0'

cuda_device='4'

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
                        first_three_digit_task_num=568,
                        task_name='MultPathTest',
                        num_institutions=2, 
                        cuda_device=cuda_device,
                        overwrite_nnunet_datadirs=True,
                        verbose=False)


# Now train on each collaborator
for task in ['Task568_MultPathTest', 'Task569_MultPathTest']:
    train_on_task(task=task, current_epoch=1)


#############################################################################################################




print("##############################################################################################\n")
print("##############################################################################################\n")
print("                            THE SINGLE TEST PASSED\n")
print("##############################################################################################\n")
print("##############################################################################################\n")