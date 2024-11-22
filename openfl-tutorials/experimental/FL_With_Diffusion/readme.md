# FL Using a Diffusion Model to Augment Data
This tutorial is for demonstrating how well score based diffusion models can be used to suplement data during federated learning.

# Basic Idea
We will do a federation across three collaborators, two of which are missing representatives of a given class. Then compare with another run where we augment these two collaborators with synthetic examples of that class. The score based diffusion model used is trained on all of CIFAR10. The idea is that potentially we have a score based diffusion model that can augment data, later we will fine tune the story.

All = airplane, automobile, frog, ship, truck

Col 0: part of All containing all frogs
Col 1:  distinct part of All with no frogs
Col 2:  distinct part of All with no frogs

Cols 0, 1, and 2 together make up all of CIFAR10 training set.

CIFAR10 test is used to test models but has only ‘All’ classes. We will pay particular focus on the accruacy for frogs, but will track others of the 'All' class as well.
