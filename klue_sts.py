import copy

import numpy as np
import torch
from datasets import load_dataset
from scipy import stats
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed


def get_score(output, label):
    score = stats.pearsonr(output, label)[0]
    return score


def train_model(epochs, device, train_dataloader, validation_dataloader, model, loss_fn, optimizer):
    model.to(device)
    loss_fn.to(device)

    best_val_score = 0
    best_model = None

    for t in range(epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            predict = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
            loss = loss_fn(predict, batch['labels'])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_pred = []
            val_label = []

            for _, val_batch in enumerate(validation_dataloader):
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                predict = model(val_batch['input_ids'], val_batch['attention_mask'], val_batch['token_type_ids'])  # validate
                loss = loss_fn(predict, val_batch['labels'])

                val_loss += loss.item()
                val_pred.extend(predict.clone().cpu().tolist())
                val_label.extend(val_batch['labels'].clone().cpu().tolist())

            val_score = get_score(np.array(val_pred), np.array(val_label))
            if best_val_score < val_score:
                best_val_score = val_score
                best_model = copy.deepcopy(model)

            print(f"\nValidation loss / cur_val_score / best_val_score : {val_loss} / {val_score} / {best_val_score}")

    return best_model


def klue_sts(args, model, tokenizer):
    # Device --
    device = args.device

    # Hyper parameter --
    set_seed(args.seed)
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    seq_max_length = args.seq_max_length

    # Dataset --
    train_dataset = load_dataset('klue', 'sts', split="train")
    validation_dataset = load_dataset('klue', 'sts', split="validation")

    # Prepare tokenizer, dataloader, model, loss function, optimizer, etc --

    def encode_input(examples):
        encoded = tokenizer(examples['sentence1'], examples['sentence2'], max_length=seq_max_length, truncation=True, padding='max_length')
        return encoded

    def format_output(examples):
        changed = {'labels': examples['labels']['label']}
        return changed

    train_dataset = train_dataset.map(encode_input)
    train_dataset = train_dataset.map(format_output)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])

    validation_dataset = validation_dataset.map(encode_input)
    validation_dataset = validation_dataset.map(format_output)
    validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])

    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    train_model(epochs, device, train_dataloader, validation_dataloader, model, loss_fn, optimizer)
