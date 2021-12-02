import torch
from transformers import BertTokenizer, BertForMaskedLM
import sys
sys.path.append(".")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("shibing624/macbert4csc-base-chinese")
model = BertForMaskedLM.from_pretrained("shibing624/macbert4csc-base-chinese")
model = model.to(device)
sentence = "你找到你最喜欢的工作，我也很高心"
tokens = ['[CLS]'] + tokenizer.tokenize(sentence) + ['[SEP]']
print(tokens)
# with torch.no_grad():
    # outputs = model(**tokenizer(texts, padding=True, return_tensors='pt').to(device))


for i in range(1, len(tokens)-1):
    tmp = tokens[:i] + ['[MASK]'] + tokens[i+1:]
    masked_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tmp)])
    segment_ids = torch.tensor([[0]*len(tmp)])
    attention_mask = torch.tensor([[1] * len(tmp)])

    outputs = model(masked_ids,attention_mask=attention_mask, token_type_ids=segment_ids)
    prediction_scores = outputs[0]
    print(tmp)
    # 打印被预测的字符
    prediction_index = torch.argmax(prediction_scores[0, i]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([prediction_index])[0]
    print(predicted_token)
