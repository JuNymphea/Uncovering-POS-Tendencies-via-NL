"""Async access to a remote chat model with a prompt-level disk cache (adapted from AxBench).

The cache is a pickle at <master_data_dir>/persist_lm_cache/{model}_{tag}_cache.pkl,
keyed by prompt text.
"""

import asyncio
import logging
import pickle
from pathlib import Path

logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

#: Accepted judge model names; catches typos in --model. Add a new judge here.
SUPPORTED_MODELS = ("gpt-5.2-2025-12-11",)


class LanguageModel(object):
    """Async chat-completion client with an optional prompt cache."""

    def __init__(self, model, client, dump_dir=None, use_cache=True,
                 cache_level="api", **kwargs):
        self.model = model
        if not any(name in model for name in SUPPORTED_MODELS):
            raise ValueError(
                f"{model} model class is not supported yet. "
                f"Known: {', '.join(SUPPORTED_MODELS)}. "
                f"Add it to SUPPORTED_MODELS in {__file__} if it is a new judge.")
        self.client = client
        self.dump_dir = None
        if dump_dir:
            self.dump_dir = Path(dump_dir) / "lm_cache"
            self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = kwargs.get("temperature", 1.0)
        self.cache_dir = None
        self.use_cache = use_cache
        self.cache_level = cache_level
        self.cache_in_mem = {}
        self.api_count = {}
        if self.use_cache:
            assert kwargs.get("master_data_dir", None), "master_data_dir is required for cache"
            self.cache_dir = Path(kwargs["master_data_dir"]) / "persist_lm_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # load cache from disk
            if kwargs.get("cache_tag", None):
                self.cache_file = self.cache_dir / f"{self.model}_{kwargs['cache_tag']}_cache.pkl"
            else:
                self.cache_file = self.cache_dir / f"{self.model}_cache.pkl"
            if self.cache_file.exists():
                with open(self.cache_file, "rb") as f:
                    self.cache_in_mem = pickle.load(f)
                logger.warning(f"loaded {len(self.cache_in_mem)} cached judgements "
                               f"from {self.cache_file}")

    def normalize(self, text):
        return text.strip()

    def _get_cache_key(self, prompt, api_count, api_name):
        if self.cache_level and self.cache_level == "prompt":
            return f"{prompt}"
        return f"{prompt}_____{api_count}_____{api_name}"

    async def chat_completion(self, client, prompt, api_name):
        # check if the prompt is cached
        api_count = self.api_count.get(api_name, 0)
        self.api_count[api_name] = api_count + 1  # increment api count
        if self.use_cache:
            cache_key = self._get_cache_key(prompt, api_count, api_name)
            if cache_key in self.cache_in_mem:
                return (self.cache_in_mem[cache_key], None)
        raw_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model, temperature=self.temperature)
        raw_completion = raw_completion.to_dict()
        completion = self.normalize(raw_completion["choices"][0]["message"]["content"])

        if self.use_cache:
            self.cache_in_mem[cache_key] = completion
        usage = raw_completion['usage']
        return (completion, usage)

    async def chat_completions(self, api_names, prompts, batch_size=32):
        """Send prompts concurrently in batches of batch_size; return completions in order."""
        # Ensure api_names is a list of appropriate length
        if not isinstance(api_names, list):
            api_names = [api_names] * len(prompts)

        # Process in batches
        all_completions = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_api_names = api_names[i:i + batch_size]

            # batched calls
            async_responses = [
                self.chat_completion(self.client, prompt, api_name)
                for prompt, api_name in zip(batch_prompts, batch_api_names)]
            raw_completions = await asyncio.gather(*async_responses)
            # post handling for current batch
            for j, (completion, usage) in enumerate(raw_completions):
                all_completions.append(completion)

        return all_completions

    def save_cache(self):
        if self.use_cache:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache_in_mem, f, protocol=pickle.HIGHEST_PROTOCOL)

    async def close(self):
        """Close the underlying HTTP client"""
        await self.client.close()
