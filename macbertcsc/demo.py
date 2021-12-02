import torch
from transformers import BertTokenizer,BertForMaskedLM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")