import json
import pandas as pd
import os
import sys
sys.path.append("your address")
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
import torch.nn.functional as F
import torch

class WHPModelForCausalLM(PreTrainedModel):
    def __init__(self, baseline_name_or_path, reinforced_name_or_path, alpha=1., config=None, **kwargs):
        if config is None:
            config = PretrainedConfig.from_pretrained(baseline_name_or_path)
        super().__init__(config)
        self.baseline = AutoModelForCausalLM.from_pretrained(baseline_name_or_path, **kwargs)
        self.reinforced = AutoModelForCausalLM.from_pretrained(reinforced_name_or_path, **kwargs)
        self.alpha = alpha


    def forward(self, input_ids=None, attention_mask=None, labels=None, return_dict=True, **kwargs):
        v_b = self.baseline(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            # return_dict=True,
                            **kwargs)
        v_r = self.reinforced(input_ids=input_ids,
                              attention_mask=attention_mask,
                              labels=labels,
                            #   return_dict=True,
                              **kwargs)
        logits = v_b.logits - self.alpha * F.relu(v_r.logits - v_b.logits)

        if not return_dict:
            return (logits,) + v_b[1:]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutput(logits=logits, loss=loss)


    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        return self.baseline.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask, **model_kwargs)
    

    def _reorder_cache(self, past, beam_idx):
        return self.baseline._reorder_cache(past, beam_idx)

def read_json(fpath: str) -> Dict | List:
    with open(fpath, 'r') as f:
        return json.load(f)


def read_text(fpath: str) -> str:
    with open(fpath, 'r') as f:
        return f.read()


def write_json(obj: Dict | List, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        return json.dump(obj, f)


def write_text(obj: str, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        return f.write(obj)


def write_csv(obj, fpath: str):
    # os.makedirs(os.path.dirname(fpath), exist_ok=True)
    pd.DataFrame(obj).to_csv(fpath, index=False)


from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
# def load_model(model_dir: str, model_name, quantize_4bit, quantize_8bit, alpha, corpus, **kwargs):
#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
#     gptq_config = GPTQConfig(bits=2, dataset = "c4", tokenizer=tokenizer)
#     model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", quantization_config=gptq_config)
#     return model

def load_model(model_dir: str, model_name, quantize_4bit, quantize_8bit, alpha, corpus, **kwargs):
    print('model_dir:', model_dir)
    if quantize_4bit==1:
        print('Load model in 4bit')
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        return AutoModelForCausalLM.from_pretrained(model_dir,
                                                    device_map='auto',
                                                    quantization_config=bnb_config,
                                                    torch_dtype=torch.bfloat16,
                                                    **kwargs)
    elif quantize_8bit==1:
        print('Load model in 8bit')
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(model_dir,
                                                    device_map='auto',
                                                    quantization_config=bnb_config,
                                                    # torch_dtype=torch.bfloat16,
                                                    **kwargs)
    else:
        print('Load model in full-precision')
        if 'whp' in model_name:
            
        return AutoModelForCausalLM.from_pretrained(model_dir,
                                                    device_map='auto',
                                                    torch_dtype=torch.bfloat16,
                                                    **kwargs)


def load_tokenizer(tokenizer_dir: str, **kwargs):
    return AutoTokenizer.from_pretrained(tokenizer_dir, **kwargs)
    