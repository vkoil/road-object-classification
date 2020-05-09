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
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #adjustable parameters
    batch_num = 5
    epochs = 20
    learningRate = 0.001
    nnet = ln5c.LeNet5C() #Check import statement on change
    lossFunc = nn.CrossEntropyLoss()
    optimizerFunc = optim.Adam(nnet.parameters(), lr=learningRate, betas=(0.9, 0.99))


    #loading datasets
    training_loader, testing_loader = acquire.load_vehicles(batch_num)
    
    #running training
    trainer.train(training_loader, optimizerFunc, lossFunc, nnet, epochs)

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

    class_correct = list(0. for i in range(len(classes)))
    class_overall = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for data in testing_loader:
            images, labels = data
            labels = helpers.label_to_index(labels)
            outputs = nnet(images)
            answer = (torch.max(outputs, 1))[1]
            correct_inc = (answer == labels).squeeze()
            for i in range(batch_num):
                label = labels[i]
                #passing 0d tensors
                if (correct_inc.dim() == 0):
                    class_correct[label] += correct_inc.item()
                else:
                    class_correct[label] += correct_inc[i].item()

                class_overall[label] += 1

    for i in range(6):
        print("Accuracy of class ", classes[i], ": ",int(100 * class_correct[i] / class_overall[i]),"%", sep = "")



if __name__ == '__main__':
    main()