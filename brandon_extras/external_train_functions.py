"""
Get a basic classifier (will use MNIST)
https://clay-atlas.com/us/blog/2021/04/22/pytorch-en-tutorial-4-train-a-model-to-classify-mnist/
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dset
from torchvision import datasets, transforms
import numpy as np
import os
import json


def load_json(path):
    with open(path, 'r') as _f:
        return json.load(_f)

def write_json(obj, path):
    with open(path, 'w') as _f:
        json.dump(obj, _f)


def get_mnist_model_path(model_dir, model_name, model_tag):
    return os.path.join(model_dir, model_name + f'_{model_tag}' + '.pt')


# You will see below that the model saves every 100 iterations and at the last iteration in an epoch. There are 938 batches in the train loader,
# So the iterations for saving are: 99, 199, 299, ..., 899, 937
    
# number batches is 938 (confirmed by getting length of loader when in memory below    
    
def mnist_model_paths(model_dir, model_name, start_epoch, epochs):
    # maybe won't end up using this
    all_epochs = range(start_epoch, start_epoch + epochs)
    paths = []
    for epoch in all_epochs:
        # This is hard coded and determined by the length of the loader () and the fact model checkpoints get saved after every 100 iterations and after the last iteration
        for iter in [99, 199, 299, 399, 499, 599, 699, 799, 899, 937]:
            paths.append(get_mnist_model_path(model_dir=model_dir, model_name=model_name, model_tag=f'epoch_{epoch}_iter_{iter}'))

    return paths


def mnist_round_to_model_startpath(model_dir, model_name, epochs, round):
    # recall the value of round (since 0 indexed) is equal to how many rounds have already trained
    epoch = round * epochs - 1   # one less than the number of epochs already trained (start epoch - 1)
    return get_mnist_model_path(model_dir=model_dir, model_name=model_name, model_tag=f'epoch_{epoch}_iter_{937}')


def mnist_round_to_model_endpath(model_dir, model_name, epochs, round):
    epoch = round * epochs - 1 + epochs   # start_epoch + epochs
    return get_mnist_model_path(model_dir=model_dir, model_name=model_name, model_tag=f'epoch_{epoch}_iter_{937}')


# Transform
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,)),]
)


# Data
trainSet = datasets.MNIST(root='MNIST', download=True, train=True, transform=transform)
testSet = datasets.MNIST(root='MNIST', download=True, train=False, transform=transform)
trainLoader = dset.DataLoader(trainSet, batch_size=64, shuffle=True)
testLoader = dset.DataLoader(testSet, batch_size=64, shuffle=False)

# Model
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=784, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
            return self.main(input)
    


# Function to hold model def only while executing, modeling external process and will transfer model state to and from disk
# This is the equivalent of the NNUnet train script
    




def train_mnist_net(config_path, data_path, model_dir, model_name, device='cuda:0', epochs=2):

    # GPU
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    print('GPU State:', device)

    mnist_net = MNISTNet().to(device)

    model_path = get_mnist_model_path(model_dir=model_dir, model_name=model_name, model_tag='init')

    mnist_net.load_state_dict(torch.load(model_path))

    # grab the config off disk
    config = load_json(config_path)
    lr = config['lr']
    momentum = config['momentum']
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(mnist_net.parameters(), lr=lr, momentum=momentum)


    metrics_over_epochs = {'loss': []}

    for epoch in range(epochs):
        running_loss = 0.0

        for times, data in enumerate(trainLoader):
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.view(inputs.shape[0], -1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Foward + backward + optimize
            outputs = mnist_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if times % 100 == 99 or times+1 == len(trainLoader):
                print('[%d/%d, %d/%d] loss: %.3f' % (epoch+1, epochs, times+1, len(trainLoader), running_loss/2000))

        checkpoint_path = get_mnist_model_path(model_dir=model_dir, model_name=model_name, model_tag=f'epoch_{epoch}')
        print(f'Saving model checkpoint to: {checkpoint_path}')

        checkpoint_dict = {
                           'model_state_dict': mnist_net.state_dict(),
                           'optimizer_state_dict': optimizer.state_dict(),
                           'loss': loss,
                           'validation': loss  # just to model a validation
        }

        torch.save(checkpoint_dict, checkpoint_path)

    metrics_over_epochs['loss'].append(running_loss/(times+1))

    # Maybe this is silly, tring to model leaving the device
    os.environ['CUDA_VISIBLE_DEVICES']=''

    print('Training Finished.')
    