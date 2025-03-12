from src.trainer import CTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
import deepspeed
from collections import defaultdict
import random


IGNORE_INDEX = -100


class KDTrainer(CTrainer):
    def __init__(self, temperature, lambda_param, ref_model, *args, **kwargs):
        super().__init__(temperature, lambda_param, ref_model, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            return_dict=True,
            output_hidden_states=False,
            output_attentions=True,
        )

        with torch.no_grad():
            target = self.ref_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                return_dict=True,
                output_hidden_states=False,
                output_attentions=True,
            )

            target_logits = target.logits

        logits = output.logits
        labels = inputs["labels"]

        # label_mask: [batch, seqlen]
        label_mask = labels.ne(IGNORE_INDEX)
        batch_size, sequence_len, vocab_size = logits.shape

        # distillation loss
        distillation_loss = self.kl_div(
            student_logits=logits, teacher_logits=target_logits
        )
        distillation_loss = distillation_loss.sum(-1) * label_mask
        distillation_loss = (
            (self.temperature**2) * distillation_loss.sum() / label_mask.sum()
        )

        """
        attention weight loss
        """
        """
        student_attention_weights/target_attention_weights:
        
        num_layers x  [batch, n_heads, seq_len, seq_len]
        num_layers = 32, n_heads = 32
        """
        nattn_layers = len(output.attentions)
        n_heads = output.attentions[0].shape[1]

        causal_mask = torch.tril(
            torch.ones(batch_size, sequence_len, sequence_len, device=label_mask.device)
        )
        mask = (label_mask.unsqueeze(-1) * causal_mask).unsqueeze(
            1
        )  # [batch, seqlen, seqlen]
        attn_loss = 0

        for i in range(nattn_layers):
            attn_loss += self.loss_fct(
                output.attentions[i].masked_fill(mask == 0, 1e-9).log(),
                target.attentions[i] * mask,
            ).sum()

        attn_loss = attn_loss / (mask[:, 0, :, 0].sum() * n_heads * nattn_layers)
        composite_loss = distillation_loss + attn_loss

        # log individual metrics
        """
        ntp: next token prediction
        """
        metrics = {
            "pred_layer_loss": distillation_loss.detach().cpu(),
            "attention_loss": attn_loss.detach().cpu(),
            "ntp_loss": output.loss.detach().cpu(),
        }

        self.store_metrics(metrics, train_eval="train")

        # overall loss
        loss = (
            1.0 - self.lambda_param
        ) * output.loss + self.lambda_param * composite_loss
        return (loss, output) if return_outputs else loss
