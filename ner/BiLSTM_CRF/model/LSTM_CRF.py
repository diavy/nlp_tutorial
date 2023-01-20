import torch
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF

class NERLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout, word2id, tag2id):
        super(NERLSTM_CRF, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id)
        self.tag_to_ix = tag2id
        self.tagset_size = len(tag2id)

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)

        #CRF
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size)

    def forward(self, x):
      #CRF
      
      embedding = self.word_embeds(x)
      outputs, hidden = self.lstm(embedding)
      outputs = self.dropout(outputs)
      outputs = self.hidden2tag(outputs)
      #CRF
      outputs = outputs.transpose(0,1)
      outputs = self.crf.decode(outputs)
      return outputs

    def log_likelihood(self, x, tags):
        embedding = self.word_embeds(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        outputs = outputs.transpose(0,1)
        tags = tags.transpose(0,1)
        return - self.crf(outputs, tags)
