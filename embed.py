import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


model = AutoModel.from_pretrained(
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True
)


def embed(documents: list[str]) -> Tensor:
    task = "根据热搜主题，返回文本的embedding"
    input_texts = [get_detailed_instruct(task, doc) for doc in documents]
    max_length = 8192
    batch_dict = tokenizer(
        input_texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    outputs = model(**batch_dict)
    embeddings = last_token_pool(
        outputs.last_hidden_state, batch_dict["attention_mask"]
    )
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


if __name__ == "__main__":
    data = json.load(Path("data/data.jsonl").open())[:5]
    embeddings = embed(data)
    print(embeddings.shape)
