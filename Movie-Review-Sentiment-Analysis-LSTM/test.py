import torch
import torch.nn as nn
import numpy as np
from train import net,Loss
from data import test_loader


test_losses = []  # track loss
num_correct = 0
h = net.init_hidden(50)
net.eval()

for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
    output, h = net(inputs, h)
    test_loss = Loss(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())

    pred = torch.round(output.squeeze())
    correct = pred.eq(labels.float().view_as(pred))  # view_as返回被视作与给定的tensor相同大小的原tensor
    correct = np.squeeze(correct.numpy())
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_losses)))
print("Test accuracy: {:.3f}".format(num_correct / len(test_loader.dataset)))











