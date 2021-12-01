import torch
from torch.utils.data import DataLoader,Dataset
from config import *


sentence=[
    # enc_input                 dec_input         dec_output
    ["How is the weather today ?", "今天天气怎么样？", "今天天气怎么样？"],
    ["Today the weather was good", "今天天气很好", "今天天气很好"],
    ["How are you ?", "你最近怎么样？","你最近怎么样？"]
]


enc_vocab = {"[PAD]":0,"[UNK]":1}
dec_vocab = {"[PAD]":0,"[UNK]":1, "[S]":2, "[E]":3}

def make_vocab(sentences):

    for sen in sentences:

        enc_input,dec_input = sen[0],sen[1]

        for word in enc_input.strip().split(" "):
            if word not in enc_vocab.copy():

                enc_vocab[word] = len(enc_vocab)

        for word in list(dec_input.strip()):
            if word not in dec_vocab.copy():
                dec_vocab[word] = len(dec_vocab)

make_vocab(sentence)
print(enc_vocab)
print(dec_vocab)

idx2word = {i: w for i, w in enumerate(dec_vocab)}
tgt_vocab_size = len(dec_vocab)

src_vocab_size = len(enc_vocab)

def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for sen in sentences:
        sen1,sen2,sen3 = sen
        sen1 = sen1.split(" ")
        if len(sen1)<max_input_len:
            enc_input = [enc_vocab.get(i,enc_vocab.get("[UNK]")) for i in sen1]+[enc_vocab.get("[PAD]")]*(max_input_len-len(sen1))
        else:
            enc_input = [enc_vocab.get(i,enc_vocab.get("[UNK]")) for i in sen1][:max_input_len]

        sen2 = ["[S]"]+list(sen2.strip())
        sen3 = list(sen3.strip())

        if len(sen2)<max_output_len:
            dec_input = [dec_vocab.get(i,enc_vocab.get("[UNK]")) for i in sen2]+[dec_vocab.get("[PAD]")]*(max_output_len-len(sen2))
        else:
            dec_input = [dec_vocab.get(i,dec_vocab.get("[UNK]")) for i in sen2][:max_output_len]

        if len(sen3) < max_output_len-1:
            sen3+=["[E]"]
            print(sen3)
            dec_output = [dec_vocab.get(i, enc_vocab.get("[UNK]")) for i in sen3] + [dec_vocab.get("[PAD]")] * (
                        max_output_len - len(sen3))
        else:
            dec_output = [dec_vocab.get(i, dec_vocab.get("[UNK]")) for i in sen3][:max_output_len-1]+[dec_vocab.get("[E]")]



        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)
    return torch.LongTensor(enc_inputs),\
           torch.LongTensor(dec_inputs),\
           torch.LongTensor(dec_outputs)

enc_inputs, dec_inputs, dec_outputs = make_data(sentence)


class MyDataSet(Dataset):

    def __init__(self,enc_inputs,dec_inputs,dec_outputs):

        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs


    def __len__(self):

        return self.enc_inputs.shape[0]


    def __getitem__(self, index):

        return self.enc_inputs[index],self.dec_inputs[index],self.dec_outputs[index]


loader = DataLoader(MyDataSet(enc_inputs,dec_inputs,dec_outputs),batch_size=3, shuffle=True)
