import torch
import torch.nn as nn


class DSSM(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size=100,
                 hidden_size=128):
        super(DSSM, self).__init__()

        self.char_embedding = nn.Embedding(vocab_size, embedding_size)

        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(0.2)

    def forward(self, char_p, char_q):
        p_embedding = self.char_embedding(char_p.long())
        q_embedding = self.char_embedding(char_q.long())

        p = torch.tanh(self.fc1(p_embedding))
        q = torch.tanh(self.fc1(q_embedding))
        p = self.dropout(p)
        q = self.dropout(q)
        p = self.fc2(p)
        q = self.fc2(q)

        p = torch.mean(p, dim=1)
        q = torch.mean(q, dim=1)

        cosine = torch.cosine_similarity(p, q)
        cosine[cosine < 0] = 0

        return cosine
