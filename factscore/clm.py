# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import time
import json
import re
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer, AutoTokenizer

from factscore.utils import convert_model_to_int8_on_gpu
from factscore.lm import LM
from factscore.few_shot_examples import q1, r1, q2, r2, q3, r3, q4, r4, q5, r5


class Mistral_LM(LM):
    def __init__(self, model_dir="mistralai/Mistral-7B-Instruct-v0.2", cache_file=None):
        
        self.model_dir = model_dir
        self.save_interval = 100
        if cache_file:
            super().__init__(cache_file)

    def load_model(self):
        
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, device_map="auto", load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, device_map="auto", torch_dtype=torch.float16)
        


        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    # for generating atomic facts
    def generate(self, q, max_new_tokens=256,
                  do_sample=True, temperature=0.7, top_p=0.95):
        
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        self.add_n += 1
            
        
        messages_to_pass = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": r1},
        {"role": "user", "content": q2},
        {"role": "assistant", "content": r2},
        {"role": "user", "content": q3},
        {"role": "assistant", "content": r3},
        {"role": "user", "content": q4},
        {"role": "assistant", "content": r4},
        {"role": "user", "content": q5},
        {"role": "assistant", "content": r5},
        {"role": "user", "content": q}
        ]

        encodeds = self.tokenizer.apply_chat_template(messages_to_pass, return_tensors="pt").to("cuda")

        generated_ids = self.model.generate(encodeds, pad_token_id=self.tokenizer.eos_token_id,
                                             max_new_tokens=max_new_tokens, do_sample=do_sample, 
                                             temperature=temperature, top_p=top_p)
        decoded = self.tokenizer.batch_decode(generated_ids)
        final_ans = decoded[0]
        final_ans = re.split(r"\[/INST\]", final_ans)[-1]
        final_ans = final_ans.replace("</s>", "")
        return final_ans
    
    # for fact checking generate only one token and its logit
    def _generate(self, prompt, max_new_tokens=1):
        
        messages = [{"role": "user", "content": prompt}]

        tokens = self.tokenizer.apply_chat_template(messages, return_tensors="pt",
                                                    add_generation_prompt=True).to("cuda")
        
        out = self.model.generate(tokens, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=max_new_tokens, 
                     output_scores=True, return_dict_in_generate=True)
        
        return out["scores"][0].detach().cpu().numpy()
            

