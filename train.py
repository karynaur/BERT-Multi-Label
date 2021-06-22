import tez
import torch
from transformers import (AdamW,
                          BertModel,
                          get_linear_schedule_with_warmup)
import torch.nn as nn

class BERT(tez.Model):
  def __init__(self, no_train_steps, num_classes = 11, lr = 1e-4):
    """
      input:
         num_classes: (int) number of classes
         no_train_steps: (int) number of training steps
         lr: (float) learning rate 
    """
    super().__init__()
    
    self.bert = BertModel.from_pretrained(
            "bert-base-uncased", return_dict = False
            )
    self.lr = lr
    self.dropout = nn.Dropout(0.3)
    self.out = nn.Linear(768, num_classes)


  def fetch_optimizer(self):
    """
        output: (torch.optimizer) A standard pytorch Optimizer
    """
    return AdamW(self.parameters(), self.lr) 
   
  def fetch_scheluder(self):
    """
         output: torch scheduler
    """
    return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps = 0,
            num_training_steps = self.no_train_steps
        ) 
  
  def losses(self, out, targets):
    return nn.CrossEntropyLoss()(out, targets)

  def forward(self, ids, mask, token_type_ids, targets = None):
    _,x = self.bert(ids, attention_mask = mask, token_type_ids = token_type_ids)
    x = self.out(self.dropout(x))

    if targets is not None:
      loss = self.losses(x, targets)
      return x, loss
    return x, 0 


