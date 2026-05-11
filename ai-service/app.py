import os
import re
from functools import lru_cache
from pathlib import Path
from typing import List

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from torch import nn
from transformers import PhobertTokenizer, RobertaConfig, RobertaModel


ROOT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT_DIR / "results" / "checkpoints_extractive" / "best_extractive_sentence_model.bin"
TOKENIZER_DIR = ROOT_DIR / "results" / "checkpoints_extractive" / "tokenizer"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="AI Summarization Service", version="1.0.0")


class SummarizeRequest(BaseModel):
  documents: List[str] = Field(default_factory=list)


class SentenceCandidate(BaseModel):
  text: str
  document_index: int
  probability: float


class SummarizeResponse(BaseModel):
  summary: str
  sentences: List[SentenceCandidate]
  meta: dict


class ExtractiveSentenceRanker(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = RobertaModel(
      RobertaConfig(
        vocab_size=64001,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=258,
        type_vocab_size=1,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        layer_norm_eps=1e-5,
      )
    )
    self.classifier = nn.Sequential(
      nn.Dropout(0.1),
      nn.Linear(768, 384),
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.Linear(384, 1),
    )

  def forward(self, input_ids, attention_mask):
    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    logits = self.classifier(cls_embedding).squeeze(-1)
    return logits, cls_embedding


@lru_cache(maxsize=1)
def load_artifacts():
  checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
  model = ExtractiveSentenceRanker()
  model.load_state_dict(checkpoint["model_state_dict"])
  model.to(DEFAULT_DEVICE)
  model.eval()
  tokenizer = PhobertTokenizer.from_pretrained(str(TOKENIZER_DIR))
  return model, tokenizer, checkpoint["config"], checkpoint


def split_sentences(text: str) -> List[str]:
  normalized = re.sub(r"\s+", " ", text).strip()
  if not normalized:
    return []
  raw_sentences = re.split(r"(?<=[\.\!\?\…])\s+|(?<=\n)\s*", normalized)
  sentences = [sentence.strip(" -\n\t") for sentence in raw_sentences]
  return [sentence for sentence in sentences if len(sentence.split()) >= 4]


def build_candidates(documents: List[str]):
  candidates = []
  for doc_index, document in enumerate(documents):
    for position, sentence in enumerate(split_sentences(document)):
      candidates.append(
        {
          "text": sentence,
          "document_index": doc_index,
          "position": position,
        }
      )
  return candidates


def batched(iterable, batch_size: int):
  for index in range(0, len(iterable), batch_size):
    yield iterable[index:index + batch_size]


def encode_candidates(candidates: List[dict], tokenizer, model, max_len: int):
  texts = [candidate["text"] for candidate in candidates]
  all_probs = []
  all_embeddings = []

  with torch.no_grad():
    for batch in batched(texts, 16):
      encoded = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
      )
      encoded.pop("token_type_ids", None)
      encoded = {key: value.to(DEFAULT_DEVICE) for key, value in encoded.items()}
      logits, embeddings = model(**encoded)
      probs = torch.sigmoid(logits).detach().cpu().numpy()
      all_probs.append(probs)
      all_embeddings.append(embeddings.detach().cpu().numpy())

  return np.concatenate(all_probs), np.concatenate(all_embeddings)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
  denominator = np.linalg.norm(a) * np.linalg.norm(b)
  if denominator == 0:
    return 0.0
  return float(np.dot(a, b) / denominator)


def score_candidates(candidates: List[dict], probabilities: np.ndarray, embeddings: np.ndarray, config: dict):
  doc_count = max((candidate["document_index"] for candidate in candidates), default=-1) + 1
  doc_seen = [0] * max(doc_count, 1)
  centroid = embeddings.mean(axis=0) if len(embeddings) else np.zeros((768,), dtype=np.float32)

  for index, candidate in enumerate(candidates):
    candidate["probability"] = float(probabilities[index])
    candidate["embedding"] = embeddings[index]
    position_bonus = max(0.0, 1.0 - (candidate["position"] * 0.12)) * config.get("POSITION_WEIGHT", 0.06)
    centrality_bonus = max(0.0, cosine_similarity(candidate["embedding"], centroid)) * config.get("CENTRALITY_WEIGHT", 0.16)
    doc_bonus = (1.0 / (1 + doc_seen[candidate["document_index"]])) * config.get("DOC_COVERAGE_BONUS_WEIGHT", 0.32)
    candidate["score"] = candidate["probability"] + position_bonus + centrality_bonus + doc_bonus
    doc_seen[candidate["document_index"]] += 1


def redundancy_ratio(a: str, b: str) -> float:
  tokens_a = set(a.lower().split())
  tokens_b = set(b.lower().split())
  union = tokens_a | tokens_b
  if not union:
    return 0.0
  return len(tokens_a & tokens_b) / len(union)


def select_sentences(candidates: List[dict], config: dict):
  selected = []
  max_sentences = int(config.get("MAX_SUMMARY_SENTENCES", 8))
  max_words = int(config.get("MAX_SUMMARY_WORDS", 220))
  threshold = float(config.get("POSITIVE_THRESHOLD", 0.24))
  redundancy_limit = float(config.get("MAX_REDUNDANCY_JACCARD", 0.48))
  mmr_lambda = float(config.get("MMR_LAMBDA", 0.68))

  candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
  current_words = 0
  covered_docs = set()

  while candidates and len(selected) < max_sentences and current_words < max_words:
    best_index = None
    best_value = -1e9

    for index, candidate in enumerate(candidates):
      if candidate["probability"] < threshold and selected:
        continue

      if any(redundancy_ratio(candidate["text"], picked["text"]) > redundancy_limit for picked in selected):
        continue

      if current_words + len(candidate["text"].split()) > max_words:
        continue

      redundancy_penalty = 0.0
      if selected:
        redundancy_penalty = max(
          cosine_similarity(candidate["embedding"], picked["embedding"]) for picked in selected
        )

      doc_bonus = 0.0 if candidate["document_index"] in covered_docs else config.get("DOC_COVERAGE_BONUS_WEIGHT", 0.32)
      mmr_value = (mmr_lambda * candidate["score"]) - ((1.0 - mmr_lambda) * redundancy_penalty) + doc_bonus

      if mmr_value > best_value:
        best_value = mmr_value
        best_index = index

    if best_index is None:
      break

    chosen = candidates.pop(best_index)
    selected.append(chosen)
    current_words += len(chosen["text"].split())
    covered_docs.add(chosen["document_index"])

  if not selected and candidates:
    selected = [candidates[0]]

  return sorted(selected, key=lambda item: (item["document_index"], item["position"]))


@app.on_event("startup")
def startup_event():
  load_artifacts()


@app.get("/health")
def health():
  _, _, _, checkpoint = load_artifacts()
  return {
    "status": "ok",
    "device": DEFAULT_DEVICE,
    "model_path": str(CHECKPOINT_PATH),
    "val_f1": checkpoint["val_f1"],
    "val_rouge": checkpoint["val_rouge"],
  }


@app.post("/api/summarize", response_model=SummarizeResponse)
def summarize(payload: SummarizeRequest):
  documents = [document.strip() for document in payload.documents if document and document.strip()]
  if not documents:
    return SummarizeResponse(summary="", sentences=[], meta={"message": "No documents provided"})

  model, tokenizer, config, checkpoint = load_artifacts()
  candidates = build_candidates(documents)

  if not candidates:
    return SummarizeResponse(summary="", sentences=[], meta={"message": "No valid sentences extracted"})

  probabilities, embeddings = encode_candidates(candidates, tokenizer, model, int(config.get("MAX_LEN", 256)))
  score_candidates(candidates, probabilities, embeddings, config)
  selected = select_sentences(candidates, config)
  summary = " ".join(item["text"] for item in selected)

  return SummarizeResponse(
    summary=summary,
    sentences=[
      SentenceCandidate(
        text=item["text"],
        document_index=item["document_index"],
        probability=round(item["probability"], 4),
      )
      for item in selected
    ],
    meta={
      "device": DEFAULT_DEVICE,
      "candidate_count": len(candidates),
      "selected_count": len(selected),
      "threshold": checkpoint["val_best_threshold"],
    },
  )
