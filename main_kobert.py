import argparse
from datetime import datetime

import torch
from kobert_tokenizer import KoBERTTokenizer
from torch import nn
from transformers import BertModel

from klue_sts import klue_sts


class KoBertForRegression(nn.Module):
    def __init__(self, bert_model_name, hidden_size, num_labels):
        super(KoBertForRegression, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        linear_out = self.linear(bert_out)
        sigmoid_out = self.sigmoid(linear_out) * 5
        return sigmoid_out.squeeze()


def main():
    # Parser --
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='skt/kobert-base-v1', type=str)
    parser.add_argument('--hidden_size', default='768', type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seq_max_length', default=128, type=int)  # FIXME check data if 128 is enough
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4885, type=int)
    parser.add_argument('--task', default="sts", type=str)

    args = parser.parse_args()
    setattr(args, 'device', f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    setattr(args, 'time', datetime.now().strftime('%Y%m%d-%H:%M:%S'))

    print('[List of arguments]')
    for a in args.__dict__:
        print(f'{a}: {args.__dict__[a]}')

    # Hyper-hyper parameters
    model_name = args.model_name
    hidden_size = args.hidden_size
    task = args.task
    data_labels_num = 1

    # Do downstream task
    if task == "sts":
        tokenizer = KoBERTTokenizer.from_pretrained(model_name)
        kobert_model = KoBertForRegression(model_name, hidden_size, data_labels_num)
        klue_sts(args, kobert_model, tokenizer)
    else:
        print(f"There is no such task as {task}")


if __name__ == '__main__':
    main()
