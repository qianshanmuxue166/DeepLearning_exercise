import torch.nn as nn
from data import vocab_to_index

class SentimentRNN(nn.Module):
    def __init__(self,vocab_size,output_size,embedding_dim,hidden_dim,n_layers,bidirectional=True,dropout = 0.5):
        super(SentimentRNN,self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        # embedding层
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        # LSTM层
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,n_layers,dropout=dropout,batch_first=True,bidirectional=bidirectional)
        # 每一个词400维  隐藏层256  2层数的大小

        # droput层
        self.droput = nn.Dropout(0.3)
        # linear层
        if bidirectional:
            self.linear = nn.Linear(hidden_dim*2,output_size)  # 512
        else:
            self.linear = nn.Linear(hidden_dim, output_size)
        # sigmoid层
        self.sig = nn.Sigmoid()

    def forward(self,x,hidden): # hidden = (h0,c0)元组
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x.long())
        lstm_out,hidden = self.lstm(embeds,hidden)

        # dropout and fully-connected layer
        out = self.linear(self.droput(lstm_out))

        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        return sig_out,hidden   # hidden([(),()])
        # 词向量维度（74073）    1     每一个词400维  隐藏层256  2层数的大小

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        number = 1
        if self.bidirectional:
            number = 2
        # 初始化（h0,c0）
        hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_()
                  ) # h0 = (2*2,batch_size,256) c0 = (2*2,batch_size,256)

        return hidden

if __name__ == '__main__':
    vocab_size = len(vocab_to_index) + 1  # +1 for the 0 padding + our word tokens
    output_size = 1
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2
    net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    #   词向量维度（74073）    1     每一个词400维  隐藏层256  2层数的大小
    print(net)



