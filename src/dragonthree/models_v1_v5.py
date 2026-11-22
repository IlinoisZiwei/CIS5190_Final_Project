"""Passive model definitions generated from Notebook 02. Importing does not train."""
import os
import tempfile
from pathlib import Path
import copy
import hashlib
import json
import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from IPython.display import display
from .project_io import ROOT, READS, load_result, labeled_urls, text_group, install_data_guard, show_figure

def metrics_from_scores(y, score):
    """Binary metrics; caller must explicitly supply an approved development/internal panel."""
    y = np.asarray(y)
    pred = (np.asarray(score) > 0).astype(int)
    return {'n': len(y), 'correct': int((pred == y).sum()), 'accuracy': float(accuracy_score(y, pred)), 'balanced_accuracy': float(balanced_accuracy_score(y, pred)), 'macro_f1': float(f1_score(y, pred, average='macro')), 'roc_auc': float(roc_auc_score(y, score)), 'confusion_matrix_Fox_NBC': confusion_matrix(y, pred, labels=[0, 1]).tolist()}

def original_grouped_folds(texts, y, seed=42, n_splits=3):
    """Original cleaned-text grouping, not the later event/template split."""
    groups = [text_group(t) for t in texts]
    mapping = {}
    for g, label in zip(groups, y):
        if g in mapping and mapping[g] != int(label):
            raise ValueError('Conflicting source labels in the same normalized-text group')
        mapping[g] = int(label)
    keys = sorted(mapping)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in cv.split(keys, [mapping[k] for k in keys]):
        a, b = ({keys[i] for i in tr}, {keys[i] for i in va})
        yield (np.array([i for i, g in enumerate(groups) if g in a]), np.array([i for i, g in enumerate(groups) if g in b]))

def hashed_inputs(texts, num_buckets=50000):
    ids, offsets = ([], [])
    for text in texts:
        offsets.append(len(ids))
        tokens = re.findall('[a-z0-9]+', text.lower())
        ids.extend([int(hashlib.md5(t.encode()).hexdigest(), 16) % num_buckets for t in tokens] or [0])
    return (torch.tensor(ids, dtype=torch.long), torch.tensor(offsets, dtype=torch.long))

class BaselineV1(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = nn.EmbeddingBag(50000, 128, mode='mean')
        self.fc = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, ids, offsets):
        return self.fc(self.embedding(ids, offsets))

    @torch.inference_mode()
    def decision_function(self, texts):
        self.eval()
        scores = []
        for start in range(0, len(texts), 64):
            logits = self(*hashed_inputs(texts[start:start + 64]))
            scores.extend((logits[:, 1] - logits[:, 0]).numpy())
        return np.asarray(scores)

    def predict(self, texts):
        return (self.decision_function(texts) > 0).astype(int)

def fit_v1(texts, y, dev_texts, dev_y, seed=42):
    """Historical dev-selected five-epoch recipe; called only by an explicit new run."""
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model = BaselineV1()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    truth, dev_truth = (torch.tensor(y, dtype=torch.long), torch.tensor(dev_y, dtype=torch.long))
    best, best_state, history = ((-1.0, -float('inf')), None, [])
    for epoch in range(1, 6):
        model.train()
        order = torch.randperm(len(texts)).tolist()
        for start in range(0, len(order), 64):
            idx = order[start:start + 64]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(*hashed_inputs([texts[i] for i in idx])), truth[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        total_loss, correct = (0.0, 0)
        with torch.no_grad():
            for start in range(0, len(dev_texts), 64):
                logits = model(*hashed_inputs(dev_texts[start:start + 64]))
                target = dev_truth[start:start + 64]
                total_loss += nn.functional.cross_entropy(logits, target, reduction='sum').item()
                correct += int((logits.argmax(1) == target).sum())
        key = (correct / len(dev_y), -total_loss / len(dev_y))
        history.append({'epoch': epoch, 'dev_accuracy': key[0], 'dev_loss': -key[1]})
        if key > best:
            best, best_state = (key, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best_state)
    return (model, history)

class LexicalV2V3:

    def __init__(self, version='v3', C=4.0, threshold=None):
        if version not in {'v2', 'v3'}:
            raise ValueError(version)
        self.version, self.C = (version, C)
        self.scale = 1.0 if version == 'v2' else 0.776760627876633
        self.threshold = (0.0 if version == 'v2' else -0.05) if threshold is None else threshold

    def make_vectors(self):
        shared = dict(lowercase=True, strip_accents='unicode', dtype=np.float64, sublinear_tf=True)
        word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, token_pattern='(?u)\\b\\w+\\b', **shared)
        char = TfidfVectorizer(analyzer='char_wb' if self.version == 'v2' else 'char', ngram_range=(3, 5) if self.version == 'v2' else (2, 5), min_df=2, max_features=100000 if self.version == 'v2' else 120000, **shared)
        return [word, char]

    def fit(self, texts, y, sample_weight=None):
        self.vectors = self.make_vectors()
        X = sparse.hstack([v.fit_transform(texts) for v in self.vectors], format='csr')
        self.clf = LinearSVC(C=self.C, dual=True, max_iter=5000 if self.version == 'v2' else 10000, random_state=42).fit(X, y, sample_weight=sample_weight)
        return self

    def transform(self, texts):
        return sparse.hstack([v.transform(texts) for v in self.vectors], format='csr')

    def decision_function(self, texts):
        return self.clf.decision_function(self.transform(texts)) / self.scale - self.threshold

    def predict(self, texts):
        return (self.decision_function(texts) > 0).astype(int)
FUNCTION_WORDS = set('a an the and or but of to in on at for with from by as is are was were be been being it its this that these those he she they we you i his her their our your not no will would can could should has have had do does did over under after before about into out up down how what why who when where'.split())
FEATURE_NAMES = ['characters', 'words', 'mean_word_length', 'max_word_length', 'std_word_length', 'short_word_fraction', 'function_word_fraction', 'unique_word_fraction', 'apostrophes', 'commas', 'colons', 'question_marks', 'non_ascii', 'uppercase']

def numeric_features(texts):
    rows = []
    for text in texts:
        tokens = re.findall('[a-z0-9]+', text.lower())
        lengths = [len(t) for t in tokens] or [0]
        count = max(len(tokens), 1)
        rows.append([len(text), len(tokens), np.mean(lengths), max(lengths), np.std(lengths), sum((len(t) <= 3 for t in tokens)) / count, sum((t in FUNCTION_WORDS for t in tokens)) / count, len(set(tokens)) / count, text.count("'") + text.count('’'), text.count(','), text.count(':'), text.count('?'), sum((ord(c) > 127 for c in text)), sum((c.isupper() for c in text))])
    return np.asarray(rows, dtype=np.float64)

def structure_matrix(lexical, numeric, scaler, interaction):
    z = scaler.transform(numeric)
    return sparse.hstack([lexical, sparse.csr_matrix(z * 0.5), lexical.multiply((np.clip(z[:, 0], -3, 3) * interaction)[:, None])], format='csr')

def normalize_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-10)

def local_encoder_components(pretrained_dir):
    from transformers import AutoModel, AutoTokenizer
    p = Path(pretrained_dir)
    if not p.is_dir():
        raise FileNotFoundError('Supply the fixed-revision local pretrained directory.')
    tok = AutoTokenizer.from_pretrained(p, local_files_only=True, clean_up_tokenization_spaces=False)
    encoder = AutoModel.from_pretrained(p, local_files_only=True, add_pooling_layer=False)
    return (tok, encoder)

@torch.inference_mode()
def frozen_embeddings(texts, pretrained_dir, layer=-1):
    tok, encoder = local_encoder_components(pretrained_dir)
    encoder.eval()
    chunks = []
    for start in range(0, len(texts), 64):
        t = tok(texts[start:start + 64], padding=True, truncation=True, max_length=64, return_tensors='pt')
        output = encoder(**t, output_hidden_states=True)
        hidden = output.hidden_states[layer]
        mask = t['attention_mask'].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        chunks.append(nn.functional.normalize(pooled, p=2, dim=1).numpy())
    return np.concatenate(chunks)

class FineMiniLM(nn.Module):

    def __init__(self, pretrained_dir):
        super().__init__()
        self.tokenizer, self.encoder = local_encoder_components(pretrained_dir)
        self.head = nn.Linear(384, 1)

    def forward(self, ids, mask):
        hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        e = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.head(nn.functional.normalize(e, p=2, dim=1)).squeeze(1)

    @torch.inference_mode()
    def score(self, texts):
        self.eval()
        out = []
        for start in range(0, len(texts), 64):
            t = self.tokenizer(texts[start:start + 64], padding=True, truncation=True, max_length=64, return_tensors='pt')
            out.append(self(t['input_ids'], t['attention_mask']).numpy())
        return np.concatenate(out)

def fine_tune_minilm(texts, y, pretrained_dir, seed, frozen):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = FineMiniLM(pretrained_dir)
    init = LogisticRegression(C=10.0, solver='liblinear', max_iter=2000, random_state=42).fit(frozen, y)
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy(init.coef_).float())
        model.head.bias.copy_(torch.from_numpy(init.intercept_).float())
    tokens = model.tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
    optimizer = torch.optim.AdamW([{'params': model.encoder.parameters(), 'lr': 2e-05}, {'params': model.head.parameters(), 'lr': 0.001}], weight_decay=0.01)
    targets = torch.tensor(y, dtype=torch.float32)
    total_steps, step = (4 * math.ceil(len(y) / 32), 0)
    for epoch in range(1, 4):
        model.train()
        order = rng.permutation(len(y))
        for start in range(0, len(order), 32):
            idx = order[start:start + 32]
            length = int(tokens['attention_mask'][idx].sum(1).max())
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens['input_ids'][idx, :length], tokens['attention_mask'][idx, :length])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            for group, rate in zip(optimizer.param_groups, [2e-05, 0.001]):
                group['lr'] = rate * max(0.1, 1 - step / total_steps)
    return model.eval()

class EnsembleV4:

    def __init__(self, pretrained_dir):
        self.pretrained_dir = pretrained_dir
        self.scales = [0.9255566836299168, 0.9241684716764983]
        self.neural_scale = 3.1162887335392746

    def fit(self, texts, y):
        torch.set_num_threads(4)
        self.vectors = LexicalV2V3('v3').make_vectors()
        L = sparse.hstack([v.fit_transform(texts) for v in self.vectors], format='csr')
        N = numeric_features(texts)
        self.scaler = StandardScaler().fit(N)
        self.lexical = [LinearSVC(C=4.0, dual=True, max_iter=30000, random_state=42).fit(structure_matrix(L, N, self.scaler, interaction), y) for interaction in [1.0, 0.25]]
        frozen = frozen_embeddings(texts, self.pretrained_dir)
        self.neural = [fine_tune_minilm(texts, y, self.pretrained_dir, seed, frozen) for seed in [42, 43]]
        return self

    def lexical_features(self, texts):
        return sparse.hstack([v.transform(texts) for v in self.vectors], format='csr')

    def decision_function(self, texts):
        L, N = (self.lexical_features(texts), numeric_features(texts))
        lexical = np.mean([model.decision_function(structure_matrix(L, N, self.scaler, interaction)) / scale for model, interaction, scale in zip(self.lexical, [1.0, 0.25], self.scales)], axis=0)
        neural = np.mean([m.score(texts) for m in self.neural], axis=0) / self.neural_scale
        return 0.75 * lexical + 0.25 * neural - -0.1

    def predict(self, texts):
        return (self.decision_function(texts) > 0).astype(int)

class HybridV5:

    def __init__(self, fitted_v4, pretrained_dir):
        self.v4, self.pretrained_dir = (fitted_v4, pretrained_dir)

    def fit(self, texts, y):
        self.vectors = LexicalV2V3('v3').make_vectors()
        L = sparse.hstack([v.fit_transform(texts) for v in self.vectors], format='csr')
        N = numeric_features(texts)
        self.scaler = StandardScaler().fit(N)
        E = frozen_embeddings(texts, self.pretrained_dir, layer=3)
        X = sparse.hstack([structure_matrix(L, N, self.scaler, 0.25), sparse.csr_matrix(1.5 * E)], format='csr')
        self.clf = LinearSVC(C=4.0, dual=True, max_iter=20000, random_state=42).fit(X, y)
        return self

    def decision_function(self, texts):
        L = sparse.hstack([v.transform(texts) for v in self.vectors], format='csr')
        E = frozen_embeddings(texts, self.pretrained_dir, layer=3)
        X = sparse.hstack([structure_matrix(L, numeric_features(texts), self.scaler, 0.25), sparse.csr_matrix(1.5 * E)], format='csr')
        raw = self.clf.decision_function(X)
        return 0.5 * raw / 1.005554992263655 + 0.5 * self.v4.decision_function(texts) - -0.05

    def predict(self, texts):
        return (self.decision_function(texts) > 0).astype(int)

def fixed_recipe_oof(factory, texts, y, seed=42):
    """For a deliberate future training run only; no execution in default review mode."""
    y = np.asarray(y)
    score = np.zeros(len(y))
    for tr, va in original_grouped_folds(texts, y, seed):
        model = factory().fit([texts[i] for i in tr], y[tr])
        score[va] = model.decision_function([texts[i] for i in va])
    return score
