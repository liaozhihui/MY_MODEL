from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch
file_path = "./ResumeNER/train.char.bmes"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def prepare_copus(file_path):

    word_to_id = {}
    tag_to_id = {"<START>":0,"<STOP>":1}
    word_list = []
    tag_list = []

    with open(file_path, "r", encoding="utf-8") as fopen:
        words = []
        tags = []
        for line in fopen:
            if line.strip():
                word, tag = line.strip().split(" ")
                words.append(word)
                tags.append(tag)
                if word not in word_to_id:
                    word_to_id[word]=len(word_to_id)

                if tag not in tag_to_id:
                    tag_to_id[tag] = len(tag_to_id)
            else:
                word_list.append(words)
                tag_list.append(tags)
                words = []
                tags = []
        tag_to_id["<pad>"]=len(tag_to_id)
    word_to_id["<UNK>"]=len(word_to_id)
    return word_to_id, tag_to_id, word_list, tag_list


def validator(model,prepare_sequence,word_to_ix,tag_to_ix,dev_word_list,dev_tag_list):
    model.eval()
    sentence_in, lengths, idx_sort = prepare_sequence(dev_word_list, word_to_ix)
    targets = [torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long,device=device) for tags in dev_tag_list]
    targets = pad_sequence(targets, batch_first=True)[idx_sort]
    loss = model.neg_log_likelihood(sentence_in.to(device), targets.to(device), lengths)
    return loss.item()

class NerDataset(Dataset):

    def __init__(self,file_path):

        self.word_to_id, self.tag_to_id, self.word_list, self.tag_list = prepare_copus(file_path)

        assert len(self.word_list) == len(self.tag_list)

        self.length = len(self.word_list)


    def __getitem__(self, index):

        return self.word_list[index],self.tag_list[index]

    def __len__(self):

        return self.length


def collate_fn(batch):
    input, target = zip(*batch)
    return list(input),list(target)

if __name__ == '__main__':

    nerdataset = NerDataset(file_path)

    dataLoader = DataLoader(dataset=nerdataset, batch_size=2, shuffle=True, collate_fn=collate_fn)


    for data in dataLoader:
        x,y=data
        print(x)
        print(y)

