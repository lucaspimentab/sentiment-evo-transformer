from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


class Vocabulary:
    def __init__(self, counter: Counter, min_freq: int) -> None:
        self.stoi: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.stoi:
                self.stoi[token] = len(self.stoi)

        self.itos: List[str] = [""] * len(self.stoi)
        for token, idx in self.stoi.items():
            self.itos[idx] = token

    @property
    def pad_index(self) -> int:
        return self.stoi["<pad>"]

    @property
    def unk_index(self) -> int:
        return self.stoi["<unk>"]

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi.get(token, self.unk_index) for token in tokens]

    def __len__(self) -> int:
        return len(self.stoi)


def build_vocabulary(texts: Iterable[str], min_freq: int) -> Vocabulary:
    counter: Counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    return Vocabulary(counter, min_freq)


def numericalise(vocab: Vocabulary, text: str) -> List[int]:
    tokens = tokenize(text)
    if not tokens:
        return [vocab.unk_index]
    return vocab.encode(tokens)


class TweetDataset(Dataset):
    def __init__(self, sequences: Sequence[List[int]], labels: Sequence[int]) -> None:
        self.sequences = list(sequences)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[List[int], int]:
        return self.sequences[index], self.labels[index]


def collate_batch(
    batch: Sequence[Tuple[List[int], int]],
    pad_index: int,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    padded_sequences: List[List[int]] = []
    for seq in sequences:
        seq = seq[:max_length]
        padding = max_length - len(seq)
        if padding > 0:
            seq = seq + [pad_index] * padding
        padded_sequences.append(seq)
    inputs = torch.tensor(padded_sequences, dtype=torch.long)
    targets = torch.tensor(labels, dtype=torch.long)
    return inputs, targets


def load_splits(
    processed_dir: Path,
    validation_split: float,
    max_samples: Optional[int],
    random_state: int,
    min_freq: int,
) -> Tuple[
    Tuple[List[List[int]], List[int]],
    Tuple[List[List[int]], List[int]],
    Vocabulary,
]:
    depressed_path = processed_dir / "depressed_tweets.csv"
    normal_path = processed_dir / "non_depressed_tweets.csv"

    depressed_texts = pd.read_csv(depressed_path)["tweet"].fillna("").astype(str)
    normal_texts = pd.read_csv(normal_path)["tweet"].fillna("").astype(str)

    if max_samples is not None:
        per_class = max_samples // 2
        depressed_texts = depressed_texts.sample(
            n=min(per_class, len(depressed_texts)),
            random_state=random_state,
            replace=False,
        )
        normal_texts = normal_texts.sample(
            n=min(per_class, len(normal_texts)),
            random_state=random_state,
            replace=False,
        )

    samples = [(text, 1) for text in depressed_texts] + [(text, 0) for text in normal_texts]
    random.Random(random_state).shuffle(samples)

    if len(samples) < 2:
        raise ValueError("Not enough samples to perform training and validation.")

    split_idx = int(len(samples) * (1.0 - validation_split))
    split_idx = max(1, min(split_idx, len(samples) - 1))

    train_samples = samples[:split_idx]
    valid_samples = samples[split_idx:]

    train_texts = [text for text, _ in train_samples]
    train_labels = [label for _, label in train_samples]
    valid_texts = [text for text, _ in valid_samples]
    valid_labels = [label for _, label in valid_samples]

    vocab = build_vocabulary(train_texts, min_freq=min_freq)
    train_sequences = [numericalise(vocab, text) for text in train_texts]
    valid_sequences = [numericalise(vocab, text) for text in valid_texts]

    return (train_sequences, train_labels), (valid_sequences, valid_labels), vocab


def create_loader(
    sequences: List[List[int]],
    labels: List[int],
    batch_size: int,
    pad_index: int,
    max_length: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TweetDataset(sequences, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_batch(batch, pad_index, max_length),
    )


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
        pad_index: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 2),
        )
        self.max_length = max_length

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = inputs.shape
        positions = torch.arange(seq_len, device=inputs.device).unsqueeze(0).expand(batch_size, seq_len)
        embeddings = self.embedding(inputs) + self.position_embedding(positions)
        padding_mask = inputs.eq(self.embedding.padding_idx)
        encoded = self.encoder(embeddings, src_key_padding_mask=padding_mask)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == targets).sum().item()
        total_examples += targets.size(0)

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return average_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pad_index: int,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    true_positive = true_negative = false_positive = false_negative = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            total_loss += loss.item() * targets.size(0)
            total_examples += targets.size(0)

            predictions = logits.argmax(dim=1)
            true_positive += ((predictions == 1) & (targets == 1)).sum().item()
            true_negative += ((predictions == 0) & (targets == 0)).sum().item()
            false_positive += ((predictions == 1) & (targets == 0)).sum().item()
            false_negative += ((predictions == 0) & (targets == 1)).sum().item()

    if total_examples == 0:
        return 0.0, {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "specificity": 0.0,
        }

    total_loss /= total_examples
    accuracy = (true_positive + true_negative) / max(total_examples, 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
    }
    return total_loss, metrics


def train_model(
    config: Dict[str, object],
    save_path: Optional[Path] = None,
    verbose: bool = True,
    device: Optional[torch.device] = None,
    min_freq: int = 2,
) -> Dict[str, float]:
    processed_dir = Path(config["processed_dir"])
    validation_split = float(config.get("validation_split", 0.2))
    max_samples = config.get("max_samples")
    random_state = int(config.get("random_state", 42))

    torch.manual_seed(random_state)
    random.seed(random_state)

    (train_seq, train_labels), (valid_seq, valid_labels), vocab = load_splits(
        processed_dir=processed_dir,
        validation_split=validation_split,
        max_samples=max_samples if max_samples is None else int(max_samples),
        random_state=random_state,
        min_freq=min_freq,
    )

    batch_size = int(config.get("batch_size", 64))
    max_length = int(config.get("max_length", 128))

    train_loader = create_loader(
        train_seq,
        train_labels,
        batch_size=batch_size,
        pad_index=vocab.pad_index,
        max_length=max_length,
        shuffle=True,
    )
    valid_loader = create_loader(
        valid_seq,
        valid_labels,
        batch_size=batch_size,
        pad_index=vocab.pad_index,
        max_length=max_length,
        shuffle=False,
    )

    model = TransformerClassifier(
        vocab_size=len(vocab),
        max_length=max_length,
        embedding_dim=int(config.get("embedding_dim", 128)),
        num_heads=int(config.get("num_heads", 4)),
        num_layers=int(config.get("num_layers", 2)),
        feedforward_dim=int(config.get("feedforward_dim", 256)),
        dropout=float(config.get("dropout", 0.1)),
        pad_index=vocab.pad_index,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    criterion = nn.CrossEntropyLoss()

    epochs = int(config.get("epochs", 5))
    best_metrics: Dict[str, float] = {}
    best_accuracy = -1.0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, metrics = evaluate(model, valid_loader, criterion, device, vocab.pad_index)

        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs} "
                f"| train_loss {train_loss:.4f} | train_acc {train_accuracy:.4f} "
                f"| val_loss {valid_loss:.4f} | val_acc {metrics['accuracy']:.4f} "
                f"| val_f1 {metrics['f1']:.4f}"
            )

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_metrics = dict(metrics)
            best_metrics["val_loss"] = valid_loss
            best_metrics["train_loss"] = train_loss
            best_metrics["train_accuracy"] = train_accuracy

            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "vocab": vocab.stoi,
                        "config": dict(config),
                        "metrics": best_metrics,
                    },
                    save_path,
                )

    return best_metrics
