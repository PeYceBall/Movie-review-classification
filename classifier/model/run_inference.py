import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import dropbox
import gc

# модель приходится скачивать динамически из-за ограничений heroku
dbx = dropbox.Dropbox("NqD9Rbe_98AAAAAAAAAADBocoMCGD9Nk4L4OIKc4iywEvKDSspcGNGukFoYl1jUe")
model_path = 'classifier/model/bert2.pt'
with open(model_path, "wb") as f:
    metadata, res = dbx.files_download(path="/bert2.pt")
    f.write(res.content)

del metadata, res
gc.collect()

device = 'cpu'
model = torch.load(model_path, map_location=device)
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def classify_comment(text, max_len=256):
    # квантили распределения логарифма вероятности,
    # по которым будет выставлен рейтинг
    quantiles = np.array([-9.15513935e+00, -8.93880043e+00, -8.27962961e+00,
                          -5.53425903e+00, -3.35847884e-01, -1.01976395e-03,
                          -2.50339508e-04, -2.18868256e-04, -1.99317932e-04])
    model.eval()
    res = tokenizer.encode_plus(text, add_special_tokens=True,
                                max_length=max_len,
                                pad_to_max_length=True,
                                truncation=True,
                                return_attention_mask=True,
                                return_tensors='pt')
    # батч состоит из одного элемента
    X_batch = res['input_ids']  # .unsqueeze(0)
    attention_mask = res['attention_mask']  # .unsqueeze(0)
    logits = model(X_batch, token_type_ids=None,
                   attention_mask=attention_mask)[0]
    # положительный комментарий - 1, отрицательный - 0
    sentiment = logits.max(1)[1].detach().cpu().numpy().item()
    # логарифм вероятности того, что комментарий положительный
    logprob = F.log_softmax(logits, dim=1)[:, 1].item()
    # оценка комментария - номер наибольшей квантили,
    # которую превышает логарифм вероятности
    score = np.sum(logprob >= quantiles) + 1

    return sentiment, score
