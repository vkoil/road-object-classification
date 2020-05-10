import Resources.helpers as helpers
import torch

def train(training_loader, optimizerFunc, lossFunc, nnet, epochs):
    epoch_plot = []
    loss_plot = []
    accuracy_plot = []
    for n in range(epochs):  # loop over the dataset multiple times
        training_loss = 0.0
        correct = 0
        overall = 0
        for i, data in enumerate(training_loader, 0):
            
            # obtaining inputs
            inputs, labels = data
            labels = helpers.label_to_index(labels)  # Note: To match the label numbers to the number of classes.

            # zeroing the gradients
            optimizerFunc.zero_grad()

            # forwards propagation
            outputs = nnet(inputs)
            loss = lossFunc(outputs, labels)
            # backwards propagation
            loss.backward()
            # update weights
            optimizerFunc.step()

            #checking accuracy
            answer = torch.max(outputs, 1)
            overall += labels.size(0)
            correct += (answer[1] == labels).sum().item()
            
            # printing loss results
            training_loss += loss.item()
            if i % 100 == 0:  # limiting prints
                print('[%d, %5d] loss: %.3f' %(n + 1, i, training_loss / 100))
                training_loss = 0.0
        accuracy_plot.append(100*correct/overall)
        loss_plot.append(training_loss)
        epoch_plot.append(n + 1)
    print()
    return (epoch_plot, loss_plot, accuracy_plot)

    #PATH = './cifar_net.pth'
    #torch.save(lenet5.state_dict(), PATH)