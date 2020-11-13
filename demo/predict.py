import torch
from text_classification.data_process import seq2index, padding_seq
import os
from config import GPU_NUM

path = os.path.dirname(__file__)

if torch.cuda.is_available():
    device = torch.device(f'cuda:{GPU_NUM}')
else:
    device = torch.device('cpu')

model = torch.load(path + '/text_cnn.p').to(device)
model.eval()


def classification_predict(s):
    s = seq2index(s)
    if torch.cuda.is_available():
        s = torch.from_numpy(padding_seq([s])).cuda().long()
    else:
        s = torch.from_numpy(padding_seq([s])).long()

    out = model(s)
    return out.cpu().data.numpy()


if __name__ == '__main__':
    while 1:
        s = input('句子：')
        print(classification_predict(s))
