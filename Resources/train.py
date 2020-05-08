import Resources.helpers as helpers

def train(training_loader, optimizerFunc, lossFunc, nnet, epochs):
    for n in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(training_loader, 0):
            
            # obtaining inputs; data is formatted as a list of [inputs, labels]
            inputs, labels = data
            labels = helpers.label_to_index(labels)  # Note: To match the label numbers to the number of classes.

            # zeroing the gradients
            optimizerFunc.zero_grad()

            # forward + backward + optimize
            outputs = nnet(inputs)  # our LeNet output
            loss = lossFunc(outputs, labels)
            loss.backward()
            optimizerFunc.step()

            # printing results
            running_loss += loss.item()
            if i % 50 == 0:  # limiting prints
                print('[%d, %5d] loss: %.3f' %(n + 1, i + 1, running_loss / 50))
                running_loss = 0.0

    print('Finished Training')

    #PATH = './cifar_net.pth'
    #torch.save(lenet5.state_dict(), PATH)