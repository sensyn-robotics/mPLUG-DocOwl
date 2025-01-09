import torch
import os
from transformers import AutoTokenizer, AutoModel
from icecream import ic
import time
import subprocess
            
class DocOwlInfer():
    def __init__(self, ckpt_path):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False)
        self.model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto')
        self.model.init_processor(tokenizer=self.tokenizer, basic_image_size=504, crop_anchors='grid_12')
        
    def inference(self, images, query):
        messages = [{'role': 'USER', 'content': '<|image|>'*len(images)+query}]
        answer = self.model.chat(messages=messages, images=images, tokenizer=self.tokenizer)
        return answer

# get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
images = [
        f'{current_dir}/examples/dourohashishihousho_1_p104.png',
    ]

# Free GPU memory
torch.cuda.empty_cache()

docowl = DocOwlInfer(ckpt_path='mPLUG/DocOwl2')

answer = docowl.inference(images, query='how can we compute 衝撃係数 for 鋼端 case？')
torch.cuda.empty_cache()