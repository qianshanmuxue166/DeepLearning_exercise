import numpy as np
from string import punctuation


############################Data Preprocessing#####################
with open("data/reviews.txt","r",encoding="utf-8") as f:
    reviews = f.read()

with open("data/labels.txt","r",encoding="utf-8") as f:
    labels = f.read()

reviews = reviews.lower()
all_text = ''.join([c for c in reviews if c not in punctuation])
text_split = all_text.split("\n") # 列表(一句话一个元素)

all_text = ''.join(text_split)
words = all_text.split()
# print(words[:200])

########################## Encoding the words #################
from collections import Counter
counts = Counter(words)
vocab = sorted(counts,key=counts.get,reverse=True)
vocab_to_index = {word:index for index,word in enumerate(vocab,1)}

text_index = [] # [[句子中所有单词的index],[],[],...]
for sentence in text_split:
    text_index.append([vocab_to_index[word] for word in sentence.split()])

# print(len(vocab_to_index))
# print(text_index[0])
# print(len(text_index)) # 25001

####################### Encoding the labels#####################3
label_split = labels.split("\n")
encoded_labels = np.array([1 if label == "positive" else 0 for label in label_split])

text_len = Counter([len(x) for x in text_index])
# print(text_len) # {132: 185, 130: 185}
# print(text_len[0]) # 1只有一个句子长度为0
# print(max(text_len)) # 句子中单词数目最长的为2514

non_zero_index = [index for index,sentence in enumerate(text_index) if len(sentence)!=0]
text_index = [text_index[index] for index in non_zero_index] # [[句子中所有单词的index],[],[],...]
encoded_labels = np.array([encoded_labels[index] for index in non_zero_index])
# print(len(text_index)) # 25000

#################### Padding sequences#######################
seq_len = 200
from tensorflow.python.keras import preprocessing
features = np.zeros((len(text_index),200),dtype=int)
features = preprocessing.sequence.pad_sequences(text_index,200)
# print(features.shape) # (25000, 200)
# print(features[:2,:])

#################### Training, Test划分 #######################
split_frac = 0.8

split_idx = int(len(features)*split_frac)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x)*0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

#################### DataLoaders and Batching #######################
import torch
from torch.utils.data import DataLoader,TensorDataset

train_data = TensorDataset(torch.from_numpy(train_x),torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

batch_size = 50
train_loader = DataLoader(train_data,shuffle=True,batch_size=batch_size)
valid_loader = DataLoader(valid_data,shuffle=True,batch_size=batch_size)
test_loader = DataLoader(test_data,shuffle=True,batch_size=batch_size)


# sample_x, sample_y = iter(train_loader).next()
# print(sample_x.size())  # batch_size, seq_length  torch.Size([50, 200])
# print(sample_x)

# print(torch.from_numpy(train_x).size()) # [20000, 200]

