import pandas as pd
import torch
from dataset import BERTDataset
from train import BERT
import random
import tez

prompts = pd.read_csv('data/all_prompts.csv')
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


id = prompts['promptId'].values
question = prompts['prompt_question'].values
prompt = {}
for i in range(4):
    prompt[id[i]] = question[i]
    
    
essay = train['essay'].values
targets = train['evaluator_rating'].values
prompts = [prompt[i] for i in train['promptId'].values]

temp = list(zip(essay, targets, prompts))
random.shuffle(temp)
del essay, prompts, targets
essay, targets, prompts = zip(*temp)

assert len(essay) == len(targets) ==len(prompts)

train_dataset = BERTDataset(prompts[:1000], essay[:1000], targets[:1000], max_len=512)
valid_dataset = BERTDataset(prompts[1000:], essay[1000:], targets[1000:], max_len=512)

bs = 32
epochs = 10

ntrain_steps = int(len(essay) / bs * epochs)
model = BERT(no_train_steps = ntrain_steps)
es = tez.callbacks.EarlyStopping(monitor="valid_loss", patience=3, model_path = 'model.bin')

model.fit(train_dataset, valid_dataset, epochs=epochs, device = "cpu", train_bs = bs, callbacks=[es])