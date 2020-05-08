"""
List of Helper functions used.

1. silu(conv_nd) - A modified activation function which takes a convolution layer object and applies it to a sigmoid function.
    This result is then returned

2. item_reriever(dataset, is_cifar10) - Takes a dataset object, and a boolean which specifies if the dataset being passed is CIFAR-10.
    It then returns a list containing indices of a dataset which have the wanted labels.

3. label_to_index(tensor_labels) - Takes a tensor of tensors containing single integers. 
    the integers are mapped to 0-6 values, and then returned as tensors of the same format. 
"""
import torch


def silu(conv_nd):
    return conv_nd * torch.sigmoid(conv_nd)



def item_retriever(dataset, is_cifar10):  # function which  takes a dataset and returns the indices of all the wanted classes.
    indices = []
    length = dataset.__len__()
    # CIFAR 10 pathway.
    if is_cifar10:
        for i in range(length):
            target = dataset.__getitem__(i)
            if (target[1] == 1):
                indices.insert(0, i)  # for separating automobiles into the first half
            elif (target[1] == 9):
                indices.append(i)  # for separating trucks into the second half
        resize = int(len(indices) / 20) # 20 chosen to let only 600 images (500 training, 100 testing) from each class through.
        indices = indices[:resize] + indices[-resize:]
        return indices
    # CIFAR 100 pathway.
    else:
        desired_classes = [8, 13, 48, 58]  # numbers that correspond to the class in cifar100 - bicycle, bus, motorcycle, pickup truck
        # CIFAR100 classes are ordered alphabetically from 0-99. Could try classes streetcar(81), tractor(89), or train(90)
        for i in range(length):
            target = dataset.__getitem__(i)
            if (target[1] in desired_classes):
                indices.append(i)
        return indices



# This function maps the irregular label numbers of CIFAR10 and CIFAR100 combined, to ordered indices which fit the number of classes.
# Using the labels on their own will throw index errors because the fully connected layer of the nn is supposed to have 6.
def label_to_index(tensor_labels):
    look_up_table = {1: 0, 9: 1, 8: 2, 13: 3, 48: 4, 58: 5}
    for i in range(len(tensor_labels)):
        tensor_labels[i] = torch.Tensor([look_up_table[int(tensor_labels[i])]])
    return tensor_labels