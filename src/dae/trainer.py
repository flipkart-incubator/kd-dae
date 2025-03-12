import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List, Union, Any
import json
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from functools import wraps
import bz2
import pickle
import functools
import pandas as pd
from copy import deepcopy
import deepspeed
from src.trainer import CTrainer


IGNORE_INDEX = -100


class DAETrainer(CTrainer):
    def __init__(
        self, temperature, lambda_param, ref_model, ref_model_domain, *args, **kwargs
    ):
        super().__init__(temperature, lambda_param, ref_model, *args, **kwargs)
        self.ref_model_domain = ref_model_domain

        if self.is_deepspeed_enabled:
            self.ref_model_domain = self._prepare_deepspeed(self.ref_model_domain)
        else:
            self.ref_model_domain = self.accelerator.prepare_model(
                self.ref_model_domain, evaluation_mode=True
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            return_dict=True,
        )

        with torch.no_grad():
            # public reference model
            target = self.ref_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                return_dict=True,
            )

            target_logits = target.logits

            # domain reference model
            domain_target = self.ref_model_domain(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                return_dict=True,
            )

            domain_target_logits = domain_target.logits

        logits = output.logits
        labels = inputs["labels"]

        # label_mask: [batch, seqlen]
        label_mask = labels.ne(IGNORE_INDEX)
        batch_size, sequence_len, vocab_size = logits.shape

        # is_domain_flag: [batch]
        is_domain_flag = inputs["is_domain"]
        domain_mask = is_domain_flag.ne(0)
        public_mask = is_domain_flag.ne(1)

        ensemble_target_logits = torch.cat(
            [domain_target_logits[domain_mask], target_logits[public_mask]]
        )
        ensemble_base_logits = torch.cat([logits[domain_mask], logits[public_mask]])
        label_mask = torch.cat([label_mask[domain_mask], label_mask[public_mask]])

        # distillation loss
        distillation_loss = self.kl_div(
            student_logits=ensemble_base_logits, teacher_logits=ensemble_target_logits
        )
        distillation_loss = distillation_loss.sum(-1) * label_mask
        distillation_loss = (
            (self.temperature**2)
            * distillation_loss.sum()
            / label_mask.sum()  # batch_size
        )

        # log individual metrics
        """
        ntp: next token prediction
        """
        metrics = {
            "pred_layer_loss": distillation_loss.detach().cpu(),
            "ntp_loss": output.loss.detach().cpu(),
        }

        self.store_metrics(metrics, train_eval="train")

        # overall loss
        loss = (
            1.0 - self.lambda_param
        ) * output.loss + self.lambda_param * distillation_loss
        return (loss, output) if return_outputs else loss


def get_trainer_class(trainer: str = "KLD"):
    return DAETrainer
