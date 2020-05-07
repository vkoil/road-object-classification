import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np



batch_num = 4
Epoch = 20
learningRate = 0.001
classes = ["Car", "Truck", "Bicycle", "Bus", "Motorcycle", "Pickup"]



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
        resize = int(len(indices) / 20)  # this needs to be done to ensure to only get 600 images (500 training and 100 testinfg) from each class.
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





#<<<<<<<<<<<<<<<<<<<< Data Acquisition >>>>>>>>>>>>>>>>>>>>

transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])# For converting data into normalised tensors in the range [-1,1].

# Obtaining CIFAR-10 and CIFAR-100
training_set10 = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transformer,download=False)  # change to true if required.
testing_set10 = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transformer, download=False)
training_set100 = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transformer, download=False)
testing_set100 = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transformer, download=False)

# Filtering classes and merging data from both CIFAR sets.
training_set10 = torch.utils.data.Subset(training_set10, item_retriever(training_set10, True))
training_set100 = torch.utils.data.Subset(training_set100, item_retriever(training_set100, False))
training_set = torch.utils.data.ConcatDataset([training_set10, training_set100])

testing_set10 = torch.utils.data.Subset(testing_set10, item_retriever(testing_set10, True))
testing_set100 = torch.utils.data.Subset(testing_set100, item_retriever(testing_set100, False))
testing_set = torch.utils.data.ConcatDataset([testing_set10, testing_set100])

training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_num, shuffle=True)
testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_num)




# <<<<<<<<<<<<<<<<<<<< Model Section >>>>>>>>>>>>>>>>>>>>

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(20 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 110)
        self.fc3 = nn.Linear(110, 50)
        self.fc4 = nn.Linear(50, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


lenet5 = LeNet()
lossFunc = nn.CrossEntropyLoss()
optimizerFunc = optim.Adam(lenet5.parameters(), lr=learningRate, betas=(0.9, 0.99))




# <<<<<<<<<<<<<<<<<<<< Training Section >>>>>>>>>>>>>>>>>>>> 

for epoch in range(Epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(training_loader, 0):
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels = label_to_index(labels)  # Note: To match the label numbers to the number of classes.

        # zero the parameter gradients
        optimizerFunc.zero_grad()

        # forward + backward + optimize
        outputs = lenet5(inputs)  # our LeNet output
        loss = lossFunc(outputs, labels)
        loss.backward()
        optimizerFunc.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 0:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0

print('Finished Training')

#PATH = './cifar_net.pth'
#torch.save(lenet5.state_dict(), PATH)





# <<<<<<<<<<<<<<<<<<<< Testing Section >>>>>>>>>>>>>>>>>>>> 

correct = 0
total = 0
with torch.no_grad():
    for data in testing_loader:
        images, labels = data
        labels = label_to_index(labels)
        outputs = lenet5(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

class_correct = list(0. for i in range(6))
class_total = list(0. for i in range(6))
with torch.no_grad():
    for data in testing_loader:
        images, labels = data
        labels = label_to_index(labels)
        outputs = lenet5(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(batch_num):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(6):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

