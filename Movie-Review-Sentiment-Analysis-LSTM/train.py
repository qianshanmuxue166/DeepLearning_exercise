import torch
import torch.nn as nn
import numpy as np
from model import SentimentRNN
from data import vocab_to_index,train_loader,valid_loader

vocab_size = len(vocab_to_index) + 1  # +1 for the 0 padding + our word tokens
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2
net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)


optim = torch.optim.Adam(net.parameters(),lr=0.001)
Loss = nn.BCELoss()
clip=5 # gradient clipping

net.train()
for epoch in range(4):
    h = net.init_hidden(50)
    for batch_idx,(inputs,labels) in enumerate(train_loader):

        h = tuple([each.data for each in h])

        output,h = net(inputs,h)
        loss = Loss(output.squeeze(),labels.float())
        optim.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optim.step()

        if batch_idx%100 == 0:
            val_h = net.init_hidden(50)
            val_losses = []

            net.eval()
            for inputs, labels in valid_loader:
                val_h = tuple([each.data for each in val_h])
                output, val_h = net(inputs, val_h)
                val_loss = Loss(output.squeeze(), labels.float())
                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(epoch+1, 4),
                  "Step: {}...".format(batch_idx),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))

















