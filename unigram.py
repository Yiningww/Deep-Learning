import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random

# Construct a corpus
corpus = "关 于 王 怡 宁 是 猪 这 件 事 我 有 几 点 想 说".split()
vocab = sorted(set(corpus)) # ['宁', '怡', '是', '猪', '王']
word2idx = {word: idx for idx, word in enumerate(vocab)} # {'宁': 0, '怡': 1, '是': 2, '猪': 3, '王': 4}
idx2word = {idx: word for word, idx in word2idx.items()} # {0: '宁', 1: '怡', 2: '是', 3: '猪', 4: '王'}
vocab_size = len(vocab)

# Construct training data (x,y): Use the previous word to predict the next word_to_idx

# training_data = []
# for idx, word in enumerate(corpus):
#     if idx == len(corpus) - 1:
#         break
#     x = word_to_idx[corpus[idx]]
#     y = word_to_idx[corpus[idx+1]]
#     training_data.append((x, y))

training_data = [(word2idx[corpus[i]], word2idx[corpus[i+1]]) for i in range(len(corpus)-1)]
# Transfer to tensor
x_train = torch.tensor([x for x, _ in training_data], dtype = torch.long)
y_train = torch.tensor([y for _, y in training_data], dtype = torch.long)

class UnigramNN(nn.Module):
    def __init__(self, vocab_size):
        super(UnigramNN, self).__init__()
        self.emb = nn.Linear(vocab_size, 32)
        self.relu1 = nn.ReLU()
        self.MLP_up = nn.Linear(32, 16)
        self.MLP_down = nn.Linear(16, vocab_size)

    def forward(self, x):
        one_hot = torch.nn.functional.one_hot(x, num_classes = vocab_size).float() # x:15 numbers, one hot: (15, vocab_size)
        x = self.emb(one_hot)
        x = self.MLP_up(x)
        x = self.MLP_down(x)
        # out = self.linear(one_hot) #n.Linear(vocab_size, vocab_size)(one_hot)
        return x

model = UnigramNN(vocab_size)
print(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(x_train) # torch.Size([15, 16]), 15 is batch size, 16 is
    training_loss = criterion(outputs, y_train)
    training_loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {training_loss.item():.4f}")

count = 0
for test in range(1000):
    with torch.no_grad():
        test_word = vocab[random.randint(0, vocab_size - 1)]
        test_idx = torch.tensor([word2idx[test_word]])
        output = model(test_idx)
        probs = torch.softmax(output, dim=1).squeeze()
        top_idx = torch.argmax(probs).item()
        count += sum(1 if x==test_idx and y==top_idx else 0 for (x,y) in training_data)
        print(f'\nTest {test + 1}, Given "{test_word}", predicted next word is "{idx2word[top_idx]}"')

print(f"overall accuracy:{count/1000}")
