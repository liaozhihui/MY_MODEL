from NerDataLoader import prepare_copus, NerDataset, collate_fn, DataLoader
from batch_bilstm_crf import *

file_path = "./ResumeNER/train.char.bmes"

word_to_ix, tag_to_ix, word_list, tag_list = prepare_copus(file_path)
ix_to_tag = dict(zip(tag_to_ix.values(),tag_to_ix.keys()))
nerdataset = NerDataset(file_path)

dataLoader = DataLoader(dataset=nerdataset, batch_size=64, shuffle=True, drop_last=True, collate_fn=collate_fn)

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

for i in range(30):
    print(f"epoch {i+1} start............")
    j=1
    for dataiter in dataLoader:
        model.zero_grad()
        x_trains, tag_trains = dataiter
        sentence_in, lengths, idx_sort = prepare_sequence(x_trains, word_to_ix)
        targets = [torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long) for tags in tag_trains]
        targets = pad_sequence(targets, batch_first=True)[idx_sort]
        loss = model.neg_log_likelihood(sentence_in, targets, lengths)
        loss.backward()
        optimizer.step()
        print(f"batch_{j} finished")
        j+=1

with torch.no_grad():
    print(tag_list[0])
    precheck_sent,lenghts,_ = prepare_sequence([word_list[0]], word_to_ix)
    result = model(precheck_sent,lenghts,mode="dev")
    print([ix_to_tag[ix] for ix in result])




