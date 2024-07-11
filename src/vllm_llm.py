from dataclasses import dataclass
from enum import Enum
from typing import List, Union, Dict 

import time
import vllm
from vllm import LLM, SamplingParams

def load_vllm_pipeline(model_id: str, device: str, gpus: int, max_allowed_memory_in_gb: int, mock: bool = False, gpu_memory_utilization: float = 0.9, max_model_len: int = 30000):
    """Loads the VLLM pipeline for the LLM"""
    if mock or model_id == "mock":
        raise NotImplementedError("Mock pipeline is not implemented")

    try:
        # Attempt to initialize the LLM with new parameters
        llm = LLM(
            model=model_id, 
            tensor_parallel_size=gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len
        )
        llm.llm_engine.tokenizer.eos_token_id = 128009
        return llm
    except Exception as e:
        print(f"Error loading the VLLM pipeline within {max_allowed_memory_in_gb}GB: {e}")
        raise e


class vLLMPipeline:
    def __init__(
        self,
        model_id: str,
        llm_max_allowed_memory_in_gb: int,
        device: str = "cuda",
        gpus: int = 1,
        mock: bool = False,
        gpu_memory_utilization: float = 0.9,  # Add this parameter
        max_model_len: int = 30000  # Add this parameter
    ):
        self.llm = load_vllm_pipeline(
            model_id, 
            device, 
            gpus, 
            llm_max_allowed_memory_in_gb, 
            mock,
            gpu_memory_utilization,  # Pass this parameter
            max_model_len  # Pass this parameter
        )
        self.mock = mock
        self.gpus = gpus
        self.tokenizer = self.llm.llm_engine.tokenizer

    def __call__(self, composed_prompt: str, **model_kwargs: Dict) -> str:
        if self.mock:
            raise NotImplementedError("Mock pipeline is not implemented")

        # Compose sampling params
        temperature = model_kwargs.get("temperature", 0.8)
        top_p = model_kwargs.get("top_p", 0.95)
        max_tokens = model_kwargs.get("max_tokens", 256)

        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
        output = self.llm.generate(composed_prompt, sampling_params, use_tqdm=True)
        response = output[0].outputs[0].text
        return response


class vLLM_LLM:
    def __init__(
        self,
        llm_pipeline: vLLMPipeline,
        system_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        self.llm_pipeline = llm_pipeline
        self.system_prompt = system_prompt
        self.model_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
        }

        # Keep track of generation data using messages and times
        self.messages: List[Dict[str, str]] = [{"content": self.system_prompt, "role": "system"}]
        self.times: List[float] = [0]
        self._role_template = {
            "system": "âsystem\n{{{{ {} }}}}â",
            "user": "âuser\n{{{{ {} }}}}â",
            "assistant": "âassistant\n{{{{ {} }}}}â",
            "end": "âassistantâ",
        }

    def query(
        self,
        message: str,
        role: str = "user",
        disregard_system_prompt: bool = False,
    ) -> str:
        # Adds the message to the list of messages for tracking purposes, even though it's not used downstream
        messages = self.messages + [{"content": message, "role": role}]

        t0 = time.time()
        response = self.forward(messages=messages)

        self.messages = messages
        self.messages.append({"content": response, "role": "assistant"})
        self.times.extend((0, time.time() - t0))

        return response

    def _make_prompt(self, messages: List[Dict[str, str]]) -> str:
        composed_prompt: List[str] = []

        for message in messages:
            role = message["role"]
            if role not in self._role_template:
                continue
            content = message["content"]
            composed_prompt.append(self._role_template[role].format(content))

        # Adds final tag indicating the assistant's turn
        composed_prompt.append(self._role_template["end"])
        return "".join(composed_prompt)

    def forward(self, messages: List[Dict[str, str]]) -> str:
        # make composed prompt from messages
        composed_prompt = self._make_prompt(messages)
        response = self.llm_pipeline(composed_prompt, **self.model_kwargs)

        #print(f"{self.__class__.__name__} generated the following output:\n{response}")

        return response


if __name__ == "__main__":
    # Example usage
    llm_pipeline = vLLMPipeline(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        device="cuda",
        llm_max_allowed_memory_in_gb=80,
        gpus=1,
        gpu_memory_utilization=0.95,  # Increase GPU memory utilization
        max_model_len=30000  # Set a lower max model length
    )
    llm = vLLM_LLM(llm_pipeline, system_prompt="You are a helpful AI assistant")

    message = "What is the capital of India? And I need a big excerpt of it."
    response = llm.query(message)
    print(response)

