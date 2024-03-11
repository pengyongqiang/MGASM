import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import path_conf


def electra_embed(docs, batch_size, dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ElectraForSequenceClassification.from_pretrained(path_conf.chinese_electra, num_labels=dim).to(device)
    tokenizer = ElectraTokenizer.from_pretrained(path_conf.chinese_electra)
    encoded_input = tokenizer(docs, add_special_tokens=True, padding=True, max_length=256,
                              truncation=True, return_tensors='pt').to(device)

    dataset = TensorDataset(encoded_input['input_ids'].to(device), encoded_input['attention_mask'].to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    embed = None
    progress_bar = tqdm(total=len(docs))
    with torch.no_grad():
        for batch in dataloader:
            batch_input_ids = batch[0].to(device)
            batch_attention_mask = batch[1].to(device)
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            probas = torch.nn.functional.softmax(outputs[0], dim=-1)
            if embed is None:
                embed = probas
            else:
                embed = torch.cat((embed, probas), dim=0)
            progress_bar.update(batch_size)
    return embed.cpu().detach()


