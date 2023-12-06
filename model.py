import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "TheBloke/Llama-2-7B-Chat-GPTQ"
MODEL_CACHE = "cache"

MODEL_DEST = "model"
TOKEN_DEST = "tokenizer"


def get_model(model_name: str, model_cache: str):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model


def get_tokenizer(model_name: str, model_cache: str):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, use_fast=True, cache_dir=MODEL_CACHE
    )
    return tokenizer


def load_model(model_dir: str = MODEL_DEST):
    return get_model(model_dir)


def load_tokenizer(token_dir: str = TOKEN_DEST):
    return get_tokenizer(token_dir)


if __name__ == "__main__":
    tokenizer = get_tokenizer(MODEL_NAME, MODEL_CACHE)
    tokenizer.save_pretrained(TOKEN_DEST)

    model = get_model(MODEL_NAME, MODEL_CACHE)
    model.save_pretrained(MODEL_DEST)
