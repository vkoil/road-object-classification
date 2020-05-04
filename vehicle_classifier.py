import torch
import torchvision
import torchvision.transforms as transforms


def item_retriever(dataset, is_cifar10): #function which  takes a dataset and returns the indices of all the wanted classes.
    indices = []
    length = dataset.__len__()
    #CIFAR 10 pathway.
    if is_cifar10:
        for i in range(length):
            target = dataset.__getitem__(i)
            if (target[1] == 1):
                indices.insert(0,i) # for separating automobiles into the first half
            elif (target[1] == 9):
                indices.append(i)      # for separating trucks into the second half
        resize = int(len(indices)/20) # this needs to be done to ensure to only get 600 images (500 training and 100 testinfg) from each class. 
        indices = indices[:resize] + indices[-resize:]
        return indices
    #CIFAR 100 pathway.
    else:
        desired_classes = [8,13,48,58] #numbers that correspond to the class in cifar100 - bicycle, bus, motorcycle, pickup truck
        #CIFAR100 classes are ordered alphabetically from 0-99. Could try classes streetcar(81), tractor(89), or train(90)
        for i in range(length):
            target = dataset.__getitem__(i)
            if (target[1] in desired_classes):
                indices.append(i)
        return indices


transformer = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) # For converting data into normalised tensors in the range [-1,1].

#Obtaining CIFAR-10 and CIFAR-100
training_set10 = torchvision.datasets.CIFAR10(root='./data',train = True, transform = transformer, download = False) #change to true if required.
testing_set10 = torchvision.datasets.CIFAR10(root='./data',train = False, transform = transformer, download = False)
training_set100 = torchvision.datasets.CIFAR100(root='./data',train = True, transform = transformer, download = False)
testing_set100 = torchvision.datasets.CIFAR100(root='./data',train = False, transform = transformer, download = False)

#Filtering classes and merging data from both CIFAR sets.  
training_set10 = torch.utils.data.Subset(training_set10, item_retriever(training_set10, True))
training_set100 = torch.utils.data.Subset(training_set100, item_retriever(training_set100, False))
training_set = torch.utils.data.ConcatDataset([training_set10, training_set100])

testing_set10 = torch.utils.data.Subset(testing_set10, item_retriever(testing_set10, True))
testing_set100 = torch.utils.data.Subset(testing_set100, item_retriever(testing_set100, False))
testing_set = torch.utils.data.ConcatDataset([testing_set10, testing_set100])

training_loader = torch.utils.data.DataLoader(training_set, batch_size = 5, shuffle=True)
testing_loader = torch.utils.data.DataLoader(testing_set, batch_size = 5)









