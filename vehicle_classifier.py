import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import Resources.helpers as helpers
import Resources.data_acquisition as acquire
import Resources.train as trainer
import Models.LeNet5_C as ln5c

def main():
    batch_num = 4
    epochs = 10
    learningRate = 0.001
    classes = ["Car", "Truck", "Bicycle", "Bus", "Motorcycle", "Pickup"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_loader, testing_loader = acquire.load_vehicles(batch_num)


    nnet = ln5c.LeNet5C() #Check import statement on change
    lossFunc = nn.CrossEntropyLoss()
    optimizerFunc = optim.Adam(nnet.parameters(), lr=learningRate, betas=(0.9, 0.99))
    trainer.train(training_loader, optimizerFunc, lossFunc, nnet, epochs)

    # <<<<<<<<<<<<<<<<<<<< Testing Section >>>>>>>>>>>>>>>>>>>> 

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testing_loader:
            images, labels = data
            labels = helpers.label_to_index(labels)
            outputs = nnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    class_correct = list(0. for i in range(6))
    class_total = list(0. for i in range(6))
    with torch.no_grad():
        for data in testing_loader:
            images, labels = data
            labels = helpers.label_to_index(labels)
            outputs = nnet(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(6):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))



if __name__ == '__main__':
    main()