import copy
import math
import os
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from torch import nn
from transformers import AutoTokenizer, PhobertTokenizer, RobertaConfig, RobertaModel


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_TOKENIZER_DIR = RESULTS_DIR / "checkpoints_extractive" / "tokenizer"
LEGACY_CHECKPOINT_PATH = ROOT_DIR / "phobert_outputs" / "best_model.pt"
PREFERRED_OUTPUT_DIR = ROOT_DIR / "phobert_cluster_rank_mmr_outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CFG_DEFAULTS = {
  "MODEL_NAME": "vinai/phobert-base-v2",
  "TOKENIZER_USE_FAST": False,
  "USE_EAGER_ATTENTION": True,
  "MAX_LEN": 256,
  "DROPOUT": 0.30,
  "USE_NUMERIC_FEATURES": True,
  "NUMERIC_FEATURE_COLUMNS": (
    "sent_doc_pos_norm",
    "sent_clus_pos_norm",
    "n_words_norm",
    "doc_size_norm",
  ),
  "NUMERIC_FEATURE_PROJ_DIM": 16,
  "TASK_MODE": "cluster_mmr",
  "USE_ADAPTIVE_BUDGET": False,
  "SUMMARY_MAX_SENTENCES": 5,
  "SUMMARY_MAX_WORDS": 180,
  "MIN_REQUIRED_SENTENCES": 1,
  "ADAPTIVE_MIN_SENTENCES": 1,
  "ADAPTIVE_MAX_SENTENCES": 6,
  "ADAPTIVE_MAX_WORDS": 180,
  "MMR_ALPHA": 0.65,
  "REDUNDANCY_WEIGHT": 0.30,
  "DOC_COVERAGE_WEIGHT": 0.06,
  "POSITION_BONUS_WEIGHT": 0.04,
  "CENTRALITY_WEIGHT": 0.06,
  "MIN_SENT_SCORE": 0.25,
  "MAX_REDUNDANCY_JACCARD": 0.55,
}

DECODING_STOPWORDS = set("""
va la cua co cho voi trong tren duoi mot nhung cac duoc da dang se thi ma nay do
khi nhu ve tu tai boi vi do de hon rat cung khong vao ra den sau truoc giua
nguoi viec nam ngay thang theo nhieu it lai neu nen hay hoac cung moi
ra vao len xuong lam bi boi day kia ay nham thong qua
""".split())

app = FastAPI(title="AI Summarization Service", version="2.0.0")


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


class PhoBERTSentenceClassifier(nn.Module):
  def __init__(self, dropout: float, numeric_feature_dim: int, numeric_feature_proj_dim: int):
    super().__init__()
    self.encoder = RobertaModel(RobertaConfig(
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
    ))
    hidden = self.encoder.config.hidden_size
    self.numeric_feature_dim = int(numeric_feature_dim or 0)
    self.numeric_feature_proj_dim = int(numeric_feature_proj_dim or 0)

    if self.numeric_feature_dim > 0:
      proj_dim = max(1, self.numeric_feature_proj_dim)
      self.numeric_encoder = nn.Sequential(
        nn.LayerNorm(self.numeric_feature_dim),
        nn.Linear(self.numeric_feature_dim, proj_dim),
        nn.GELU(),
        nn.Dropout(dropout * 0.5),
      )
      classifier_in = hidden + proj_dim
    else:
      self.numeric_encoder = None
      classifier_in = hidden

    self.dropout = nn.Dropout(dropout)
    self.classifier = nn.Sequential(
      nn.Linear(classifier_in, hidden // 2),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hidden // 2, 1),
    )

  def forward(self, input_ids, attention_mask=None, numeric_features=None):
    out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    cls = self.dropout(out.last_hidden_state[:, 0])

    if self.numeric_encoder is not None:
      if numeric_features is None:
        numeric_features = torch.zeros((cls.size(0), self.numeric_feature_dim), dtype=cls.dtype, device=cls.device)
      numeric_emb = self.numeric_encoder(numeric_features.to(device=cls.device, dtype=cls.dtype))
      cls = torch.cat([cls, numeric_emb], dim=1)

    return self.classifier(cls).squeeze(-1)


def cfg_ns(raw: Optional[dict]):
  merged = dict(CFG_DEFAULTS)
  raw_cfg = raw or {}
  merged.update(raw_cfg)

  adaptive_sentences = raw_cfg.get("ADAPTIVE_MAX_SENTENCES")
  adaptive_words = raw_cfg.get("ADAPTIVE_MAX_WORDS")
  if adaptive_sentences is not None:
    merged["ADAPTIVE_MAX_SENTENCES"] = adaptive_sentences
  if adaptive_words is not None:
    merged["ADAPTIVE_MAX_WORDS"] = adaptive_words

  task_mode = str(merged.get("TASK_MODE") or "cluster_mmr").strip().lower()
  merged["TASK_MODE"] = "cluster_mmr" if task_mode in {"cluster_mmr", "single_topic", "single_topic_enrichment"} else task_mode
  return SimpleNamespace(**merged)


def normalize_for_model(text: str) -> str:
  if pd.isna(text):
    return ""
  text = unicodedata.normalize("NFKC", str(text)).replace("\xa0", " ")
  return re.sub(r"\s+", " ", text).strip()


def simple_tokens(text: str) -> List[str]:
  return re.findall(r"[\w-]+", normalize_for_model(text).lower(), flags=re.UNICODE)


def count_words(text: str) -> int:
  return len(simple_tokens(text))


def jaccard_text(a: str, b: str) -> float:
  ta, tb = set(simple_tokens(a)), set(simple_tokens(b))
  return 0.0 if not ta or not tb else len(ta & tb) / max(1, len(ta | tb))


def custom_sentence_split(text: str) -> List[str]:
  text = normalize_for_model(text)
  if not text:
    return []
  for abbr in ["TP.", "TP.HCM.", "ThS.", "TS.", "PGS.", "GS.", "Mr.", "Ms.", "Dr.", "P.", "Q."]:
    text = text.replace(abbr, abbr.replace(".", "<DOT>"))
  chunks = [part.replace("<DOT>", ".").strip(" -*") for part in re.split(r"(?<=[.!?;:])\s+", text) if part and part.strip()]
  if len(chunks) <= 1:
    chunks = [part.strip().replace("<DOT>", ".") for part in re.split(r",\s+", text) if part.strip()]
  return [part for part in chunks if count_words(part) >= 2]


def make_model_input(row) -> str:
  sentence = normalize_for_model(row.get("sentence", ""))
  return f"Cau: {sentence}"


def get_numeric_feature_columns(cfg) -> List[str]:
  if not getattr(cfg, "USE_NUMERIC_FEATURES", False):
    return []
  return list(getattr(cfg, "NUMERIC_FEATURE_COLUMNS", []))


def add_numeric_features(sent_df: pd.DataFrame, cfg) -> pd.DataFrame:
  df = sent_df.copy()
  df = df.sort_values(["cluster_id", "sent_clus_pos", "doc_id", "sent_doc_pos", "raw_index"], kind="stable").reset_index(drop=True)

  cluster_size = df.groupby("cluster_id")["sentence"].transform("size").astype(float)
  doc_keys = ["cluster_id", "doc_id"]
  doc_size = df.groupby(doc_keys, dropna=False)["sentence"].transform("size").astype(float)

  doc_order = df.groupby(doc_keys, dropna=False).cumcount().astype(float)
  cluster_order = df.groupby("cluster_id", dropna=False).cumcount().astype(float)

  sent_doc_pos = pd.to_numeric(df["sent_doc_pos"], errors="coerce").fillna(-1).astype(float)
  sent_clus_pos = pd.to_numeric(df["sent_clus_pos"], errors="coerce").fillna(-1).astype(float)

  safe_doc_pos = np.where(sent_doc_pos >= 0, sent_doc_pos, doc_order)
  safe_clus_pos = np.where(sent_clus_pos >= 0, sent_clus_pos, cluster_order)

  df["sent_doc_pos_norm"] = safe_doc_pos / np.maximum(doc_size.values - 1.0, 1.0)
  df["sent_clus_pos_norm"] = safe_clus_pos / np.maximum(cluster_size.values - 1.0, 1.0)

  n_words = pd.to_numeric(df["n_words"], errors="coerce").fillna(0).astype(float)
  df["n_words_norm"] = np.clip(np.log1p(n_words) / np.log1p(80.0), 0.0, 1.0)
  df["doc_size_norm"] = np.clip(doc_size.values / np.maximum(cluster_size.values, 1.0), 0.0, 1.0)

  for col in get_numeric_feature_columns(cfg):
    values = df[col] if col in df.columns else pd.Series(0.0, index=df.index, dtype="float32")
    df[col] = pd.to_numeric(values, errors="coerce").fillna(0.0).astype("float32").clip(lower=0.0, upper=1.0)

  return df


def build_custom_df(contents, cfg, cluster_id: str = "CUSTOM_CLUSTER") -> pd.DataFrame:
  if isinstance(contents, str):
    contents = [contents]

  rows = []
  global_pos = 0
  for doc_idx, content in enumerate(contents, start=1):
    for sent_doc_pos, sentence in enumerate(custom_sentence_split(content)):
      rows.append({
        "raw_index": global_pos,
        "cluster_id": cluster_id,
        "doc_id": str(doc_idx),
        "sent_clus_pos": global_pos,
        "sent_doc_pos": sent_doc_pos,
        "sentence": sentence,
        "label": 0,
        "n_words": count_words(sentence),
      })
      global_pos += 1

  df = pd.DataFrame(rows)
  if df.empty:
    raise ValueError("Khong tach duoc cau nao tu dau vao.")

  df = add_numeric_features(df, cfg)
  df["sent_global_index"] = np.arange(len(df))
  return df


def meaningful_tokens(text: str) -> List[str]:
  return [tok for tok in simple_tokens(text) if len(tok) >= 2 and tok not in DECODING_STOPWORDS and not tok.isdigit()]


def position_bonus(row) -> float:
  sent_doc_pos = row.get("sent_doc_pos", -1)
  sent_clus_pos = row.get("sent_clus_pos", -1)
  b1 = 1.0 / (1.0 + max(0, sent_doc_pos if sent_doc_pos >= 0 else 999))
  b2 = 1.0 / math.sqrt(1.0 + max(0, sent_clus_pos if sent_clus_pos >= 0 else 999))
  return 0.7 * b1 + 0.3 * b2


def compute_centrality_scores(df: pd.DataFrame) -> pd.Series:
  sentences = df["sentence"].fillna("").astype(str).tolist()
  freq = Counter(meaningful_tokens(" ".join(sentences)))
  repeated_keywords = {token for token, count in freq.items() if count >= 2}
  if not repeated_keywords:
    repeated_keywords = {token for token, _ in freq.most_common(20)}

  def relevance(sent: str) -> float:
    toks = set(meaningful_tokens(sent))
    if not toks or not repeated_keywords:
      return 0.0
    keyword_overlap = len(toks & repeated_keywords) / max(1, len(toks))
    other_sims = [jaccard_text(sent, other) for other in sentences if other != sent]
    pairwise_centrality = float(np.mean(other_sims)) if other_sims else 0.0
    return max(keyword_overlap, pairwise_centrality)

  return pd.Series([relevance(sentence) for sentence in sentences], index=df.index, dtype="float32")


def recompute_final_score(scored_df: pd.DataFrame, cfg) -> pd.DataFrame:
  out = scored_df.copy()
  for col in ["model_score", "position_score", "centrality_score"]:
    if col not in out.columns:
      out[col] = 0.0
    out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(float)

  out["final_score"] = (
    float(cfg.MMR_ALPHA) * out["model_score"]
    + float(getattr(cfg, "POSITION_BONUS_WEIGHT", 0.04)) * out["position_score"]
    + float(getattr(cfg, "CENTRALITY_WEIGHT", 0.06)) * out["centrality_score"]
  )
  return out


def add_decoding_scores(df: pd.DataFrame, scores: np.ndarray, cfg) -> pd.DataFrame:
  out = df.reset_index(drop=True).copy()
  out["_orig_idx"] = np.arange(len(out))
  out["model_score"] = np.asarray(scores, dtype=float)
  out["position_score"] = out.apply(position_bonus, axis=1)
  out["centrality_score"] = compute_centrality_scores(out)
  return recompute_final_score(out, cfg)


def sort_selected_indices_for_reading(selected: List[int], scored_df: pd.DataFrame) -> List[int]:
  by_orig = scored_df.set_index("_orig_idx")
  return sorted(selected, key=lambda idx: (by_orig.loc[idx].get("sent_clus_pos", 10**9), str(by_orig.loc[idx].get("doc_id", "")), by_orig.loc[idx].get("sent_doc_pos", 10**9)))


def try_add_candidate(selected: List[int], selected_docs: set, selected_texts: List[str], total_words: int, row, cfg, allow_over_budget_if_empty: bool = True, loosen_redundancy: bool = False):
  sent = row["sentence"]
  n_words = int(row.get("n_words", count_words(sent)))

  if row["_orig_idx"] in selected:
    return False, total_words
  if total_words + n_words > cfg.SUMMARY_MAX_WORDS and (selected or not allow_over_budget_if_empty):
    return False, total_words

  redundancy = max([jaccard_text(sent, old) for old in selected_texts], default=0.0)
  redundancy_thr = 0.85 if loosen_redundancy else cfg.MAX_REDUNDANCY_JACCARD
  if selected and redundancy > redundancy_thr:
    return False, total_words

  selected.append(int(row["_orig_idx"]))
  selected_docs.add(row.get("doc_id"))
  selected_texts.append(sent)
  return True, total_words + n_words


def select_cluster_mmr(scored_df: pd.DataFrame, cfg) -> List[int]:
  df = scored_df.copy()
  selected, selected_texts = [], []
  selected_docs = set()
  total_words = 0
  candidate_idxs = df.index.tolist()

  while candidate_idxs and len(selected) < cfg.SUMMARY_MAX_SENTENCES:
    best_idx, best_score = None, -1e18

    for idx in candidate_idxs:
      row = df.loc[idx]
      if row["model_score"] < cfg.MIN_SENT_SCORE and len(selected) >= cfg.MIN_REQUIRED_SENTENCES:
        continue

      sent = row["sentence"]
      n_words = int(row.get("n_words", count_words(sent)))
      if total_words + n_words > cfg.SUMMARY_MAX_WORDS and selected:
        continue

      redundancy = max([jaccard_text(sent, old) for old in selected_texts], default=0.0)
      if selected and redundancy > cfg.MAX_REDUNDANCY_JACCARD:
        continue

      doc_bonus = 1.0 if row.get("doc_id") not in selected_docs else 0.0
      mmr = float(row["final_score"]) - float(cfg.REDUNDANCY_WEIGHT) * redundancy + float(cfg.DOC_COVERAGE_WEIGHT) * doc_bonus
      if mmr > best_score:
        best_idx, best_score = idx, mmr

    if best_idx is None:
      break

    added, total_words = try_add_candidate(selected, selected_docs, selected_texts, total_words, df.loc[best_idx], cfg, allow_over_budget_if_empty=True)
    candidate_idxs.remove(best_idx)
    if not added:
      continue

  min_needed = min(int(cfg.MIN_REQUIRED_SENTENCES), int(cfg.SUMMARY_MAX_SENTENCES), len(scored_df))
  if len(selected) < min_needed:
    for _, row in scored_df.sort_values("final_score", ascending=False).iterrows():
      if len(selected) >= min_needed:
        break
      added, total_words = try_add_candidate(selected, selected_docs, selected_texts, total_words, row, cfg, allow_over_budget_if_empty=True, loosen_redundancy=True)
      if not added:
        continue

  if not selected and len(scored_df) > 0:
    selected = [int(scored_df.sort_values("final_score", ascending=False).iloc[0]["_orig_idx"])]

  return sort_selected_indices_for_reading(selected, scored_df)


def clamp_value(value: int, low: int, high: int) -> int:
  return max(low, min(high, value))


def compute_adaptive_budget(scored_df: pd.DataFrame, cfg) -> Dict:
  n_sents = len(scored_df)
  n_docs = scored_df["doc_id"].nunique() if "doc_id" in scored_df.columns else 1
  total_words = int(scored_df["sentence"].fillna("").astype(str).map(count_words).sum()) if "sentence" in scored_df.columns else 0

  hard_max_sents = min(cfg.SUMMARY_MAX_SENTENCES, getattr(cfg, "ADAPTIVE_MAX_SENTENCES", cfg.SUMMARY_MAX_SENTENCES))
  hard_max_words = min(cfg.SUMMARY_MAX_WORDS, getattr(cfg, "ADAPTIVE_MAX_WORDS", cfg.SUMMARY_MAX_WORDS))

  raw_target_sents = int(round(math.sqrt(max(n_sents, 1)))) + 1
  min_required = min(hard_max_sents, getattr(cfg, "ADAPTIVE_MIN_SENTENCES", 1))
  target_sents = clamp_value(raw_target_sents, min_required, hard_max_sents)

  target_words = int(round(total_words * 0.35))
  target_words = clamp_value(target_words, max(80, target_sents * 20), hard_max_words)
  target_words = min(target_words, target_sents * 36)

  target_sents = min(target_sents, n_sents)
  min_required = min(min_required, target_sents)

  return {
    "target_sents": int(target_sents),
    "target_words": int(target_words),
    "min_required": int(min_required),
    "n_sents": int(n_sents),
    "n_docs": int(n_docs),
    "total_words": int(total_words),
  }


def make_runtime_cfg(cfg, budget: Dict):
  runtime_cfg = copy.copy(cfg)
  runtime_cfg.SUMMARY_MAX_SENTENCES = int(budget["target_sents"])
  runtime_cfg.SUMMARY_MAX_WORDS = int(budget["target_words"])
  runtime_cfg.MIN_REQUIRED_SENTENCES = int(budget["min_required"])
  runtime_cfg.RUNTIME_BUDGET = budget
  return runtime_cfg


def select_summary_indices(scored_df: pd.DataFrame, cfg):
  if getattr(cfg, "USE_ADAPTIVE_BUDGET", False):
    budget = compute_adaptive_budget(scored_df, cfg)
    runtime_cfg = make_runtime_cfg(cfg, budget)
  else:
    budget = {
      "target_sents": int(cfg.SUMMARY_MAX_SENTENCES),
      "target_words": int(cfg.SUMMARY_MAX_WORDS),
      "min_required": int(cfg.MIN_REQUIRED_SENTENCES),
      "n_sents": len(scored_df),
      "n_docs": scored_df["doc_id"].nunique() if "doc_id" in scored_df.columns else 1,
      "total_words": int(scored_df["sentence"].fillna("").astype(str).map(count_words).sum()) if "sentence" in scored_df.columns else 0,
    }
    runtime_cfg = copy.copy(cfg)
    runtime_cfg.RUNTIME_BUDGET = budget

  cfg.LAST_RUNTIME_BUDGET = budget
  return select_cluster_mmr(scored_df, runtime_cfg), "cluster_mmr"


def apply_decoder_params_to_cfg(cfg, params: Optional[Dict], verbose: bool = False):
  if not params:
    return cfg
  for key, value in params.items():
    if hasattr(cfg, key):
      setattr(cfg, key, value)
  if verbose:
    cfg.APPLIED_DECODER_PARAMS = {key: getattr(cfg, key) for key in params if hasattr(cfg, key)}
  return cfg


def current_decoder_params(cfg) -> Dict:
  return {
    "MIN_SENT_SCORE": float(getattr(cfg, "MIN_SENT_SCORE", 0.0)),
    "SUMMARY_MAX_SENTENCES": int(getattr(cfg, "SUMMARY_MAX_SENTENCES", 0)),
    "MIN_REQUIRED_SENTENCES": int(getattr(cfg, "MIN_REQUIRED_SENTENCES", 0)),
    "SUMMARY_MAX_WORDS": int(getattr(cfg, "SUMMARY_MAX_WORDS", 0)),
    "MMR_ALPHA": float(getattr(cfg, "MMR_ALPHA", 0.0)),
    "REDUNDANCY_WEIGHT": float(getattr(cfg, "REDUNDANCY_WEIGHT", 0.0)),
    "CENTRALITY_WEIGHT": float(getattr(cfg, "CENTRALITY_WEIGHT", 0.0)),
  }


def predict_scores(df: pd.DataFrame, model, tokenizer, cfg, batch_size: int = 32) -> np.ndarray:
  texts = [make_model_input(row) for _, row in df.iterrows()]
  feature_cols = get_numeric_feature_columns(cfg)
  numeric = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32").values if feature_cols else np.zeros((len(df), 0), dtype="float32")

  probs = []
  with torch.no_grad():
    for start in range(0, len(texts), batch_size):
      encoded = tokenizer(
        texts[start:start + batch_size],
        truncation=True,
        max_length=int(cfg.MAX_LEN),
        padding=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors="pt",
      )
      encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
      if feature_cols:
        encoded["numeric_features"] = torch.tensor(numeric[start:start + batch_size], dtype=torch.float32, device=DEVICE)
      probs.extend(torch.sigmoid(model(**encoded)).detach().cpu().numpy().tolist())
  return np.asarray(probs, dtype=float)


def summarize_documents(documents: List[str], model, tokenizer, cfg):
  df = build_custom_df(documents, cfg)
  scored_df = add_decoding_scores(df, predict_scores(df, model, tokenizer, cfg), cfg)
  selected_idx, decoding_mode = select_summary_indices(scored_df, cfg)
  selected_info = scored_df.set_index("_orig_idx").loc[selected_idx].reset_index()
  return {
    "summary": " ".join(df.iloc[idx]["sentence"] for idx in selected_idx),
    "selected_info": selected_info,
    "selected_indices": selected_idx,
    "scored_df": scored_df,
    "decoding_mode": decoding_mode,
    "runtime_budget": getattr(cfg, "LAST_RUNTIME_BUDGET", None),
  }


def latest_file(paths: List[Path]) -> Optional[Path]:
  existing = [path for path in paths if path.is_file()]
  if not existing:
    return None
  return max(existing, key=lambda path: path.stat().st_mtime)


def resolve_artifact_paths() -> tuple[Path, Optional[Path]]:
  preferred = PREFERRED_OUTPUT_DIR / "best_model.pt"
  candidates = [
    preferred,
    *ROOT_DIR.glob("*_outputs/best_model.pt"),
    *RESULTS_DIR.glob("**/best_model.pt"),
    *RESULTS_DIR.glob("**/best_extractive_sentence_model.bin"),
    LEGACY_CHECKPOINT_PATH,
  ]
  checkpoint_path = latest_file(candidates)
  if checkpoint_path is None:
    raise FileNotFoundError("Khong tim thay checkpoint best_model.pt hoac best_extractive_sentence_model.bin.")

  tokenizer_candidates = [
    checkpoint_path.parent / "tokenizer",
    DEFAULT_TOKENIZER_DIR,
  ]
  tokenizer_dir = next((path for path in tokenizer_candidates if path.is_dir()), None)
  return checkpoint_path, tokenizer_dir


def load_tokenizer(cfg, tokenizer_dir: Optional[Path]):
  if tokenizer_dir is not None and tokenizer_dir.is_dir():
    return PhobertTokenizer.from_pretrained(str(tokenizer_dir))

  model_name = getattr(cfg, "MODEL_NAME", CFG_DEFAULTS["MODEL_NAME"])
  use_fast = bool(getattr(cfg, "TOKENIZER_USE_FAST", False))
  try:
    return AutoTokenizer.from_pretrained(model_name, use_fast=use_fast, local_files_only=True)
  except Exception:
    if tokenizer_dir is None:
      raise
    return PhobertTokenizer.from_pretrained(str(tokenizer_dir))


@lru_cache(maxsize=1)
def load_artifacts():
  checkpoint_path, tokenizer_dir = resolve_artifact_paths()
  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  cfg = cfg_ns(checkpoint.get("cfg"))
  apply_decoder_params_to_cfg(cfg, checkpoint.get("best_decoder_params"), verbose=True)
  cfg.MIN_SENT_SCORE = 0.15

  model = PhoBERTSentenceClassifier(
    dropout=float(cfg.DROPOUT),
    numeric_feature_dim=len(get_numeric_feature_columns(cfg)),
    numeric_feature_proj_dim=int(getattr(cfg, "NUMERIC_FEATURE_PROJ_DIM", 16)),
  )
  model.load_state_dict(checkpoint["model_state_dict"])
  model.to(DEVICE)
  model.eval()

  tokenizer = load_tokenizer(cfg, tokenizer_dir)
  return model, tokenizer, cfg, checkpoint, checkpoint_path, tokenizer_dir


@app.on_event("startup")
def startup_event():
  load_artifacts()


@app.get("/health")
def health():
  _, _, cfg, checkpoint, checkpoint_path, tokenizer_dir = load_artifacts()
  return {
    "status": "ok",
    "device": DEVICE,
    "model_path": str(checkpoint_path),
    "tokenizer_path": str(tokenizer_dir) if tokenizer_dir else None,
    "model_name": getattr(cfg, "MODEL_NAME", None),
    "task_mode": getattr(cfg, "TASK_MODE", None),
    "checkpoint_decoder_params": checkpoint.get("best_decoder_params"),
    "runtime_decoder_params": current_decoder_params(cfg),
    "threshold": checkpoint.get("threshold"),
    "best_epoch": checkpoint.get("best_epoch"),
    "best_score": checkpoint.get("best_score"),
    "best_metric": checkpoint.get("best_metric"),
  }


@app.post("/api/summarize", response_model=SummarizeResponse)
def summarize(payload: SummarizeRequest):
  documents = [doc.strip() for doc in payload.documents if doc and doc.strip()]
  if not documents:
    return SummarizeResponse(summary="", sentences=[], meta={"message": "No documents provided"})

  model, tokenizer, cfg, checkpoint, checkpoint_path, tokenizer_dir = load_artifacts()
  result = summarize_documents(documents, model, tokenizer, cfg)

  return SummarizeResponse(
    summary=result["summary"],
    sentences=[
      SentenceCandidate(
        text=row["sentence"],
        document_index=int(str(row["doc_id"])) - 1 if str(row.get("doc_id", "")).isdigit() else 0,
        probability=round(float(row["model_score"]), 4),
      )
      for _, row in result["selected_info"].iterrows()
    ],
    meta={
      "device": DEVICE,
      "model_path": str(checkpoint_path),
      "tokenizer_path": str(tokenizer_dir) if tokenizer_dir else None,
      "model_name": getattr(cfg, "MODEL_NAME", None),
      "task_mode": result["decoding_mode"],
      "checkpoint_decoder_params": checkpoint.get("best_decoder_params"),
      "runtime_decoder_params": current_decoder_params(cfg),
      "candidate_count": int(len(result["scored_df"])),
      "selected_count": int(len(result["selected_indices"])),
      "threshold": checkpoint.get("threshold"),
      "runtime_budget": result["runtime_budget"],
    },
  )
