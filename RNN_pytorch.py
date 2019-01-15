import torch
import torch.autograd as autograd # torch中自动计算梯度模块
import torch.nn as nn             # 神经网络模块
import torch.nn.functional as F   # 神经网络模块中的常用功能
import torch.optim as optim       # 模型优化器模块
import one_hot_encoder as ohe
import Preprocess as pre
import numpy as np

torch.manual_seed(1)

EMBEDDING_DIM = 16026
HIDDEN_DIM = 3000
#PATH = ['BU_dataset/b19/Apr95/', 'BU_dataset/b19/Feb95/', 'BU_dataset/b19/Jan95/', 'BU_dataset/b19/Mar95/', 'BU_dataset/b19/May95/']
PATH = ['BU_dataset/b19/Apr95/']

def One_hot_encoding(path):
    raw_data, ds_size= pre.get_files(path)
    label_machine,one_hot_machine, cat_num = ohe.One_hot_encoder(path)
    raw_set =[]
    buff = []
    data_set = []
    training_set = []
    testing_set = []
    null_label = [0.0 for n in range(cat_num)]
    for sequence in raw_data:
        buff = []
        for word in sequence:
            buff.append(one_hot_machine.transform([label_machine.transform([word])]))
        #print('buffer_transform')
        raw_set.append(buff)
    for data in raw_set:
        label =[]
        sentence =[]
        for i, word in enumerate(data):
            sentence.append(word.tolist()[0])
            try:
                label.append(data[i+1].tolist()[0])
            except:
                label.append(null_label)
        data_set.append((sentence, label))
    size = len(data_set)
    training_set = data_set[:int(size*0.8)]
    testing_set = data_set[int(size*0.8):-1]
    #print(len(training_set))
    #print(len(testing_set))
    return training_set,testing_set, cat_num

def Url2vec_encoding(path):
    pass

#LSTM Model
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
    def forward(self, sentence):
        #embeds = self.word_embeddings(sentence)
        embeds = sentence
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores


def turn_FloatTensor(x):
    tensor = torch.Tensor(x)
    return autograd.Variable(tensor)
def turn_Longtensor(x):
    tensor = torch.LongTensor(x)
    return autograd.Variable(tensor)


if __name__ == "__main__":
    training_set,testing_set, cat_num = One_hot_encoding(PATH)
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, EMBEDDING_DIM, EMBEDDING_DIM)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    #Training Parts
    for epoch in range(300):
        for sentence, tags in training_set:
            model.zero_grad()
            model.hidden = model.init_hidden()
            sentence_in = turn_FloatTensor(sentence)
            buff = []
            for i in tags:
                a = np.array(i)
                buff.append(np.argmax(a))
            targets = turn_Longtensor(buff)
            tag_scores = model(sentence_in)
            loss_fn = torch.nn.CrossEntropyLoss(reduce=False, size_average=False)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
            #print('step:', epoch)
            #print('loss:',loss)
