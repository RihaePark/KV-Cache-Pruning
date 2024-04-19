import argparse
import logging

import numpy as np
import torch
import json
import tqdm 
import copy 
import re

from collections import Counter

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from datasets import load_dataset, load_metric

from modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": convert_kvcache_llama_heavy_recent
}

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_arch", type=str, default='llama')
    parser.add_argument("--model_name", type=str, default='huggyllama/llama-13b')
    parser.add_argument("--cache_dir", type=str, default='../../checkpoint/')

    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)

    parser.add_argument("--length", type=int, default=64)

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")
    set_seed(args)

    model_name = args.model_name
    config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
    config.heavy_ratio = args.heavy_ratio
    config.recent_ratio = args.recent_ratio

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir)
    model = ENABLE_Heavy_Hitter_FUNCTIONS[args.model_arch](model, config)
    model.half().eval().cuda()

    dataset = load_dataset("gsm8k", "main")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    model.eval()

    total_f1 = 0
    skip = 0
    num_batch = len(data_loader)

    with torch.no_grad() :
        for i, batch in enumerate(data_loader) :

            question = batch["question"]
            answer = batch["answer"]

            question_ids = tokenizer(question, return_tensors="pt").to(model.device)
            answer_ids = tokenizer(answer, return_tensors="pt")['input_ids'].to(model.device)
            answer_ids = answer_ids[:, 1:]
            generated_ids = model.generate(**question_ids, max_length=192, max_new_tokens=64, do_sample=True, temperature=0.6, top_p=0.9).to(model.device)
            generated_ids = generated_ids[:, question_ids['input_ids'].shape[1]:]

            question_str = tokenizer.decode(question_ids['input_ids'][0])
            generated_str = tokenizer.decode(generated_ids[0])        
            answer_str = tokenizer.decode(answer_ids[0])
            answer_str = answer_str[:answer_str.find("#")]

            while answer_str.find("<<") != -1 :
                start = answer_str.find("<<")
                end = answer_str.find(">>")
                answer_str = answer_str[:start] + answer_str[end + 2:]

            answer_str = answer_str.replace("+", " + ")
            answer_str = answer_str.replace("-", " - ")
            answer_str = answer_str.replace("*", " * ")
            answer_str = answer_str.replace("/", " / ")
            answer_str = answer_str.replace("=", " = ")

            answer_str = answer_str.split()
            generated_str = generated_str.split()

            answer_count = Counter(answer_str)
            generated_count = Counter(generated_str)

            overlap = sum(min(generated_count[w], answer_count[w]) for w in generated_count.keys() & answer_count.keys())
            
            if overlap == 0 :
                f1 = 0
            else : 
                recall = overlap / len(answer_count)
                precision = overlap / len(generated_count)      
                f1 = 2 * (recall * precision) / (recall + precision) * 100

            total_f1 = total_f1 + f1
            print(f"Batch {i}/{num_batch} score : ", f1)

        print("F1 score : ", total_f1 / (len(data_loader) - skip))

if __name__ == "__main__":
    main()
