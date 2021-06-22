import torch
import torch.nn as nn
import transformers


class BERTDataset():
  def __init__(self, prompts, texts, targets, max_len):
    """
      Input:
        texts: Pandas dataframe
        targets: Pandas dataframe
        max_len: (int) maximum number of tokens per block
    """

    self.texts = texts
    self.targets = targets
    self.prompts = prompts
    
    self.tokenizer = transformers.BertTokenizer.from_pretrained(
          "bert-base-uncased",
          do_lower_case = False
        )
    self.max_len = max_len

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    essay = f"Topic: {self.prompts[idx]}\n Essay: {self.texts[idx]}"
    score = torch.zeros(11, dtype = torch.long)
    score[int(self.targets[idx])] = 1

    inputs = self.tokenizer.encode_plus(
          essay,
          None,
          add_special_tokens = True,
          max_length = self.max_len,
          padding = "max_length",
          truncation = True
        )
    return {
        "ids" : torch.tensor(inputs["input_ids"], dtype = torch.long),
        "mask" : torch.tensor(inputs["attention_mask"], dtype = torch.long),
        "token_type_ids" : torch.tensor(inputs["token_type_ids"], dtype = torch.long),
        "targets" : score,
        } 
