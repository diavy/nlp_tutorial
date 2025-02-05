from sklearn.metrics import accuracy_score
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import fix_seed
import torch
import pandas as pd

path = 'E:\\ptm\\roberta'
tokenizer = BertTokenizer.from_pretrained(path)


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
    train_df = pd.read_csv('../data/tnews_public/train.csv')
    train_text = train_df['text'].tolist()
    train_label = train_df['label'].tolist()
    train_text = tokenizer(text=train_text,
                           return_tensors='pt',
                           truncation=True,
                           padding=True,
                           max_length=32)
    train_loader = DataLoader(BaseDataset(train_text, train_label),
                              batch_size,
                              pin_memory=True if torch.cuda.is_available() else False,
                              shuffle=False)

    dev_df = pd.read_csv('../data/tnews_public/dev.csv')
    dev_text = dev_df['text'].tolist()
    dev_label = dev_df['label'].tolist()
    dev_text = tokenizer(text=dev_text,
                         return_tensors='pt',
                         truncation=True,
                         padding=True,
                         max_length=32)

    dev_loader = DataLoader(BaseDataset(dev_text, dev_label),
                            batch_size,
                            pin_memory=True if torch.cuda.is_available() else False,
                            shuffle=False)

    return train_loader, dev_loader


# 训练模型
def train():
    fix_seed()

    train_data_loader, dev_data_loader = load_data(32)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = BertForSequenceClassification.from_pretrained(path, num_labels=4)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    best_acc = 0
    for epoch in range(5):
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
        if acc > best_acc:
            torch.save(model.state_dict(), 'teacher.bin')
            best_acc = acc


if __name__ == '__main__':
    train()
