# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .model import load_model, load_tokenizer

# from auto_gptq import AutoGPTQForCausalLM

MODEL_NAME = "TheBloke/Llama-2-70b-Chat-GPTQ"
# MODEL_BASENAME = "gptq_model-4bit--1g"
MODEL_CACHE = "cache"
# use_triton = True


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     MODEL_NAME, use_fast=True, cache_dir=MODEL_CACHE
        # )
        # model = AutoModelForCausalLM.from_pretrained(
        #     MODEL_NAME,
        #     torch_dtype=torch.float16,
        #     device_map="auto",
        # )
        self.tokenizer = load_tokenizer()
        model = load_model()
        # Pytorch 2 optimization
        self.model = torch.compile(model)

    def predict(
        self,
        prompt: str = Input(description="Prompt to send", default="Tell me about AI"),
        system_prompt: str = Input(
            description="System prompt that helps guide system behavior",
            default="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
        ),
        max_new_tokens: int = Input(
            description="Number of new tokens", ge=1, le=4096, default=512
        ),
        temperature: float = Input(
            description="Randomness of outputs, 0 is deterministic, greater than 1 is random",
            ge=0,
            le=5,
            default=0.75,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1,
            default=0.95,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it",
            ge=0,
            le=5,
            default=1.1,
        ),
        exponential_decay_start: int = Input(
            description="Number of tokens to wait before starting exponential decay.",
            default=512,
            ge=1,
            le=4096,
        ),
        exponential_decay_factor: float = Input(
            description="Decay factor for LogitProcessor exponential decay.",
            default=1.0,
            ge=1.0,
            le=10.0,
        ),
    ) -> str:
        """Run a single prediction on the model"""
        prompt_template = f"""[INST] <<SYS>>
        {system_prompt}
        <</SYS>>[/INST]
        {prompt}"""

        # input_ids = self.tokenizer(prompt_template, return_tensors="pt").input_ids.to(
        #     "cuda:0"
        # )
        exponential_decay_penalty = (exponential_decay_start, exponential_decay_factor)
        input_ids = self.tokenizer(prompt_template, return_tensors="pt")
        outputs = self.model.generate(
            inputs=input_ids,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            exponential_decay_penalty=exponential_decay_penalty,
        )
        output = self.tokenizer.decode(outputs[0])
        parts = output.split("[/INST]", 1)
        final = parts[1]

        return final
