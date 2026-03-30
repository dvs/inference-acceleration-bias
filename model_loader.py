from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)
import torch
from awq import AutoAWQForCausalLM


class ModelLoader:
    MODEL_PATHS = {
        "llama2chat": {
            "base": "meta-llama/Llama-2-7b-chat-hf",
            "awq": "ai-and-society/llama-2-7b-chat-hf-awq",
            "bitsandbytes": "meta-llama/Llama-2-7b-chat-hf",
            "kvcachequant": "meta-llama/Llama-2-7b-chat-hf",
            "wanda_struct": "ai-and-society/llama-2-7b-chat-hf-wanda-structured-2-4",
            "wanda_unstruct": "ai-and-society/llama-2-7b-chat-hf-wanda-unstruct-50",
        },
        "llama3chat": {
            "base": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "awq": "ai-and-society/llama-3.1-8B-Instruct-awq",
            "bitsandbytes": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "kvcachequant": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "wanda_struct": "ai-and-society/llama-3.1-8B-Instruct-wanda-structured-2-4",
            "wanda_unstruct": "ai-and-society/llama-3.1-8B-Instruct-wanda-unstruct-50",
        },
        "mistralchat": {
            "base": "mistralai/Mistral-7B-Instruct-v0.3",
            "awq": "ai-and-society/mistral-7B-Instruct-v0.3-awq",
            "bitsandbytes": "mistralai/Mistral-7B-Instruct-v0.3",
            "kvcachequant": "mistralai/Mistral-7B-Instruct-v0.3",
            "wanda_struct": "ai-and-society/mistral-7B-Instruct-v0.3-wanda-structured-2-4",
            "wanda_unstruct": "ai-and-society/mistral-7B-Instruct-v0.3-wanda-wanda-unstruct-50",
        },
        "qwen3": {
            "base": "Qwen/Qwen3-8B",
            "awq": "Qwen/Qwen3-8B-AWQ",
            "bitsandbytes": "Qwen/Qwen3-8B",
            "gptq": "JunHowie/Qwen3-8B-GPTQ-Int4",
        },
    }

    @classmethod
    def get_model_path(cls, model_name, deployment=None):
        base_name = model_name.lower()
        if base_name not in cls.MODEL_PATHS:
            raise ValueError(f"Unknown model: {base_name}")

        if deployment:
            method, type_ = deployment["method"], deployment["type"]
            if method in ["quantization", "pruning"]:
                return Path(cls.MODEL_PATHS[base_name][type_])
        return Path(cls.MODEL_PATHS[base_name]["base"])

    @classmethod
    def load_model(cls, model_name, deployment=None, sampling_method="greedy"):
        # returns a model, tokenizer, and generation config
        model_path = cls.get_model_path(model_name, deployment)
        cache_implementation = None
        cache_config = None

        if deployment:
            method, type_ = deployment["method"], deployment["type"]
            if method == "quantization":

                nbits = deployment.get("nbits", None)
                if type_ == "awq":
                    model, tokenizer = cls._load_awq_model(model_path, nbits=nbits)
                elif type_ == "bitsandbytes":
                    model, tokenizer = cls._load_bitsandbytes_model(
                        model_path, nbits=nbits
                    )
                elif type_ == "kvcachequant":
                    model, tokenizer = cls._load_kvcache_model(model_path)
                    cache_implementation = "quantized"
                    # currently, we only support 4-bit quantization with HQQ
                    cache_config = {
                        "backend": "HQQ",
                        "nbits": nbits,  # 4
                        "device": model.device,
                    }
                elif type_ == "gptq":
                    # Load GPTQ-quantized repo via transformers (requires gptqmodel package installed)
                    model, tokenizer = cls._load_gptq_via_transformers(model_path)

            elif method == "pruning":
                if type_ in ["wanda_struct", "wanda_unstruct"]:
                    model, tokenizer = cls._load_pruned_model(model_path)
        else:
            model, tokenizer = cls._load_base_model(model_path)

        generation_config = cls.load_generation_config(model_name, sampling_method)
        return model, tokenizer, generation_config, cache_implementation, cache_config

    @staticmethod
    def _load_awq_model(model_path, nbits):  # TODO nbits
        model = AutoAWQForCausalLM.from_quantized(model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer

    @staticmethod
    def _load_bitsandbytes_model(model_path, nbits):
        if nbits == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )

        elif nbits == 8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer

    @staticmethod
    def _load_kvcache_model(model_path):
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer

    @staticmethod
    def _load_pruned_model(model_path):
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer

    @staticmethod
    def _load_base_model(model_path):
        model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer

    @staticmethod
    def _load_gptq_via_transformers(model_path):
        """
        Load a GPTQ-quantized HF repo (e.g. JunHowie/Qwen3-8B-GPTQ-Int4)
        via transformers. This relies on the gptqmodel package being installed
        so that transformers dispatches to the GPTQ-backed model implementation.
        """
        # For Qwen-like repos trust_remote_code=True is often required
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer

    @classmethod
    def get_base_model_name(cls, model_name):
        base_name = model_name.lower()
        if base_name not in cls.MODEL_PATHS:
            raise ValueError(f"Unknown model: {base_name}")
        return cls.MODEL_PATHS[base_name]["base"]

    @classmethod
    def load_generation_config(
        cls, model_name, sampling_method="greedy"
    ) -> GenerationConfig:
        # Get the base model name
        base_name = cls.get_base_model_name(model_name)

        # Load the generation configuration for the model
        generation_config = GenerationConfig.from_pretrained(base_name)

        if sampling_method == "sampling":
            generation_config.do_sample = True
            generation_config.temperature = 1.0
            generation_config.top_k = 0
            generation_config.top_p = 1
        else:
            # Use greedy decoding
            generation_config.do_sample = False
            generation_config.temperature = None
            generation_config.top_k = None
            generation_config.top_p = None

        return generation_config
