from NerDataLoader import prepare_copus, NerDataset, collate_fn, DataLoader,validator
from batch_bilstm_crf import *
import torch
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")
print(device)
file_path = "./ResumeNER/train.char.bmes"
test_file_path = "./ResumeNER/test.char.bmes"
dev_file_path = "./ResumeNER/dev.char.bmes"
dev_word_to_ix, dev_tag_to_ix, dev_word_list, dev_tag_list = prepare_copus(dev_file_path)
word_to_ix, tag_to_ix, word_list, tag_list = prepare_copus(file_path)
test_word_to_ix, test_tag_to_ix, test_word_list, test_tag_list = prepare_copus(test_file_path)
ix_to_tag = dict(zip(tag_to_ix.values(),tag_to_ix.keys()))
nerdataset = NerDataset(file_path)

dataLoader = DataLoader(dataset=nerdataset, batch_size=512, shuffle=True, drop_last=True, collate_fn=collate_fn,num_workers=8)

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

best_val_loss = 1000
for i in range(30):
    j=1
    start_time = time.time()
    model.train()
    for dataiter in dataLoader:
        model.zero_grad()
        x_trains, tag_trains = dataiter
        sentence_in, lengths, idx_sort = prepare_sequence(x_trains, word_to_ix)
        targets = [torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long,device=device) for tags in tag_trains]
        targets = pad_sequence(targets, batch_first=True)[idx_sort]
        loss = model.neg_log_likelihood(sentence_in.to(device), targets.to(device), lengths)
        loss.backward()
        optimizer.step()
        print(f"epoch {i+1}, batch_{j}: Loss:{loss:.4f}")
        j+=1
    val_loss = validator(model,prepare_sequence,word_to_ix,tag_to_ix,dev_word_list,dev_tag_list)
    if val_loss <best_val_loss:
        torch.save(model.state_dict(), "./bi_crf.pt")
        best_val_loss = val_loss
        print(f"epoch {i+1},val_loss:{best_val_loss:.4f}")
    print("耗时:", time.time() - start_time)

with torch.no_grad():
    print(tag_list[0])
    precheck_sent,lenghts,_ = prepare_sequence([word_list[0]], word_to_ix)
    result = model(precheck_sent,lenghts,mode="dev")
    print(result)
    path = result[1]
    print([ix_to_tag[ix] for ix in list(path[0].numpy())])




