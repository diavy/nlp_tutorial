import pickle
from config import opt 
with open(opt.pickle_path, 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
        x_valid = pickle.load(inp)
        y_valid = pickle.load(inp)
print("train len:",len(x_train))
print("test len:",len(x_test))
print("valid len", len(x_valid))
print(word2id)
print(tag2id)
sentences = []
for x in x_train[:5]:
    print(x)
    sentence = ''.join([id2word[i] for i in x if i != 0])
    sentences.append(sentence)
print(sentences)
for x in y_train[:5]:
    print(x)