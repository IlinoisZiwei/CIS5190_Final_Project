"""Standalone float64 TF-IDF linear inference for the locked E2 candidates.

The exporter replaces LAYOUT with buffer dimensions. The generated model.py
requires only NumPy and PyTorch; its single checkpoint contains the vocabulary,
IDF, coefficients, and decision offset. It never fits or downloads anything.
Class order is FoxNews=0, NBC=1. Inputs are already-cleaned headline strings.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter

import numpy as np
import torch
from torch import nn

LAYOUT = []  # EXPORT_LAYOUT


def normalized_text(text):
    text = str(text).lower()
    if not text.isascii():
        text = "".join(
            char for char in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(char)
        )
    return text


def features(text, analyzer, ngram_range):
    low, high = ngram_range
    if analyzer == "word":
        tokens = re.findall(r"(?u)\b\w+\b", text)
        for size in range(low, min(high, len(tokens)) + 1):
            for start in range(len(tokens) - size + 1):
                yield " ".join(tokens[start:start + size])
    elif analyzer == "char":
        text = re.sub(r"\s\s+", " ", text)
        for size in range(low, min(high, len(text)) + 1):
            for start in range(len(text) - size + 1):
                yield text[start:start + size]
    else:
        raise ValueError("This export supports only the locked word and char blocks.")


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        for index, shape in enumerate(LAYOUT):
            self.register_buffer(
                f"metadata_{index}", torch.zeros(shape["metadata_bytes"], dtype=torch.uint8)
            )
            self.register_buffer(
                f"idf_{index}", torch.zeros(shape["features"], dtype=torch.float64)
            )
            self.register_buffer(
                f"weight_{index}", torch.zeros(shape["features"], dtype=torch.float64)
            )
        self.register_buffer("bias", torch.zeros(1, dtype=torch.float64))
        self._cache = None

    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs)
        self._cache = None
        return result

    def _decoded_blocks(self):
        if self._cache is None:
            if not LAYOUT:
                raise RuntimeError("Load the generated model.py and matching model.pt.")
            blocks = []
            for index in range(len(LAYOUT)):
                raw = getattr(self, f"metadata_{index}").detach().cpu().numpy().tobytes()
                try:
                    config = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError("Checkpoint is unloaded or does not match model.py.") from exc
                vocabulary = {term: i for i, term in enumerate(config.pop("terms"))}
                blocks.append((
                    config, vocabulary,
                    getattr(self, f"idf_{index}").detach().cpu().numpy(),
                    getattr(self, f"weight_{index}").detach().cpu().numpy(),
                ))
            self._cache = blocks
        return self._cache

    @torch.no_grad()
    def decision_function(self, batch):
        texts = list(batch)
        if not texts:
            return np.empty(0, dtype=np.float64)
        blocks = self._decoded_blocks()
        scores = np.full(len(texts), float(self.bias.item()), dtype=np.float64)
        for row, original in enumerate(texts):
            text = normalized_text(original)
            for config, vocabulary, idf, weights in blocks:
                counts = Counter(features(text, config["analyzer"], config["ngram_range"]))
                present = [(vocabulary[token], count) for token, count in counts.items()
                           if token in vocabulary]
                if not present:
                    continue
                indices = np.fromiter((item[0] for item in present), dtype=np.int64)
                values = np.fromiter((item[1] for item in present), dtype=np.float64)
                if config["binary"]:
                    values.fill(1.0)
                if config["sublinear_tf"]:
                    values = 1.0 + np.log(values)
                values *= idf[indices]
                if config["norm"] == "l2":
                    length = np.linalg.norm(values)
                    if length:
                        values /= length
                elif config["norm"] is not None:
                    raise ValueError("Only L2 or no normalization is supported.")
                scores[row] += np.dot(values, weights[indices])
        return scores

    def forward(self, batch):
        scores = torch.from_numpy(self.decision_function(batch))
        return torch.stack((-scores, scores), dim=1)

    def predict(self, batch):
        return ["NBC" if score > 0 else "FoxNews" for score in self.decision_function(batch)]


def get_model():
    return Model()
