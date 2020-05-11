import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import Resources.helpers as helpers
import Resources.data_acquisition as acquire
import Resources.train as trainer
import Models.LeNet5_D as model0

def main():
    
    #adjustable parameters
    batch_num = 10
    epochs = 20
    learningRate = 0.0005
    nnet = model0.LeNet5D()# ---> Check import statement on change, and check data acquisition and model for no. of input channels. 
    lossFunc = nn.CrossEntropyLoss()
    optimizerFunc = optim.Adam(nnet.parameters(), lr=learningRate, betas=(0.9, 0.99))


    #loading datasets
    training_loader, testing_loader = acquire.load_vehicles(batch_num)
    
    #running training and retrieving statistics 
    epoch_plot, loss_plot, accuracy_plot = trainer.train(training_loader, optimizerFunc, lossFunc, nnet, epochs)

    #set classes
    classes = ["Car", "Truck", "Bicycle", "Bus", "Motorcycle", "Pickup"]

    
    # <<<<<<<<<<<<<<<<<<<< Testing Section >>>>>>>>>>>>>>>>>>>> 

    correct = 0
    overall = 0
    with torch.no_grad():
        for data in testing_loader:
            images, labels = data
            labels = helpers.label_to_index(labels) #mapping labels to indices for consistency
            outputs = nnet(images)
            answer = (torch.max(outputs.data, 1))[1]
            overall += labels.size(0)
            correct += (answer == labels).sum().item()


    print('Network accuracy: %d %%' % (100 * correct / overall))
    print()
    print("Confusion Matrix: ")

    class_correct = list(0. for i in range(len(classes)))
    class_overall = list(0. for i in range(len(classes)))
    # for confusion matrix
    true = []
    predicted = []
    with torch.no_grad():
        for data in testing_loader:
            images, labels = data
            labels = helpers.label_to_index(labels)
            outputs = nnet(images)
            answer = (torch.max(outputs, 1))[1]
            correct_inc = (answer == labels).squeeze()
            for i in range(batch_num):
                label = labels[i]
                true.append(label)
                predicted.append(answer[i].item())
                #passing 0d tensors
                if (correct_inc.dim() == 0):
                    class_correct[label] += correct_inc.item()
                else:
                    class_correct[label] += correct_inc[i].item()

                class_overall[label] += 1
    print(metrics.confusion_matrix(true,predicted))
    print()
    print(metrics.classification_report(true,predicted,target_names= classes, digits = 4))
    print()

    for i in range(6):
        print("Accuracy of class ", classes[i], ": ",int(100 * class_correct[i] / class_overall[i]),"%", sep = "")

    plt.figure()
    plt.subplot(211)
    plt.plot(epoch_plot, loss_plot)
    plt.ylabel("Running Loss")
    plt.subplot(212)
    plt.plot(epoch_plot, accuracy_plot)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")

    plt.show()
    



if __name__ == '__main__':
    main()