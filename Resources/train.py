import Resources.helpers as helpers

def train(training_loader, optimizerFunc, lossFunc, nnet, epochs):
    for n in range(epochs):  # loop over the dataset multiple times
        training_loss = 0.0
        for i, data in enumerate(training_loader, 0):
            
            # obtaining inputs; data is formatted as a list of [inputs, labels]
            inputs, labels = data
            labels = helpers.label_to_index(labels)  # Note: To match the label numbers to the number of classes.

            # zeroing the gradients
            optimizerFunc.zero_grad()

            outputs = nnet(inputs)
            # forwards propagation
            loss = lossFunc(outputs, labels)
            # backwards propagation
            loss.backward()
            # update weights
            optimizerFunc.step()

            # printing results
            training_loss += loss.item()
            if i % 100 == 0:  # limiting prints
                print('[%d, %5d] loss: %.3f' %(n + 1, i, training_loss / 100))
                training_loss = 0.0

    print()

    #PATH = './cifar_net.pth'
    #torch.save(lenet5.state_dict(), PATH)