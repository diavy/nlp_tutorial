from sklearn.metrics import accuracy_score
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from adversarial_training.adversarial import *
from utils import fix_seed
from transformers import RobertaTokenizer


# path = ptm_path('roberta')
# tokenizer = BertTokenizer.from_pretrained(path)

# model_path = 'roberta-base'
# tokenizer = RobertaTokenizer.from_pretrained(model_path)

model_path = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_path)


class BaseDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def load_data(batch_size=32):
    train_text = []
    train_label = []
    with open('../data/sentiment/sentiment.train.data', encoding='utf-8')as file:
        for line in file.readlines():
            t, l = line.strip().split('\t')
            train_text.append(t)
            train_label.append(int(l))

    train_text = tokenizer(text=train_text,
                           return_tensors='pt',
                           truncation=True,
                           padding=True,
                           max_length=10)

    train_loader = DataLoader(BaseDataset(train_text, train_label),
                              batch_size,
                              pin_memory=True if torch.cuda.is_available() else False,
                              shuffle=False)

    dev_text = []
    dev_label = []
    with open('../data/sentiment/sentiment.valid.data', encoding='utf-8')as file:
        for line in file.readlines():
            t, l = line.strip().split('\t')
            dev_text.append(t)
            dev_label.append(int(l))

    dev_text = tokenizer(text=dev_text,
                         return_tensors='pt',
                         truncation=True,
                         padding=True,
                         max_length=10)

    dev_loader = DataLoader(BaseDataset(dev_text, dev_label),
                            batch_size,
                            pin_memory=True if torch.cuda.is_available() else False,
                            shuffle=False)

    return train_loader, dev_loader


# 训练模型
def train(num_epochs):
    fix_seed()

    train_data_loader, dev_data_loader = load_data(32)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    attack = True

    if attack:
        # at = FGM(model)
        at = FreeAT(model)

    def adversarial(data):
        optimizer.zero_grad()
        # 添加扰动
        at.attack(emb_name='embeddings.word_embeddings.weight')
        # 重新计算梯度
        adv_loss = model(input_ids=data['input_ids'].to(device),
                         attention_mask=data['attention_mask'].to(device),
                         labels=data['labels'].to(device)).loss
        # bp得到新的梯度
        adv_loss.backward()
        at.restore(emb_name='embeddings.word_embeddings.weight')

    def adversarial_free(data, m=3):
        # 备份梯度
        at.backup_grad()
        for i in range(m):
            at.attack(emb_name='embeddings.word_embeddings.weight', first_attack=i == 0)
            if i == 0:
                optimizer.zero_grad()
            else:
                at.restore_grad()
            # fp
            adv_loss = model(input_ids=data['input_ids'].to(device),
                             attention_mask=data['attention_mask'].to(device),
                             labels=data['labels'].to(device)).loss
            # bp得到新的梯度
            adv_loss.backward()
        at.restore(emb_name='embeddings.word_embeddings.weight')

    for epoch in range(num_epochs):
        print('epoch:', epoch + 1)
        pred = []
        label = []
        pbar = tqdm(train_data_loader)
        for data in pbar:
            optimizer.zero_grad()

            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device).long()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            output = outputs.logits.argmax(1).cpu().numpy()
            pred.extend(output)
            label.extend(labels.cpu().numpy())
            loss = outputs.loss
            loss.backward()

            if attack:
                # adversarial(data)
                adversarial_free(data)

            optimizer.step()

            pbar.update()
            pbar.set_description(f'loss:{loss.item():.4f}')

        acc = accuracy_score(pred, label)
        print('train acc:', acc)

        pred = []
        label = []
        for data in tqdm(dev_data_loader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device).long()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            output = outputs.logits.argmax(1).cpu().numpy()
            pred.extend(output)
            label.extend(labels.cpu().numpy())
        acc = accuracy_score(pred, label)
        print('dev acc:', acc)
        print()


if __name__ == '__main__':
    num_epochs = 5
    train(num_epochs)