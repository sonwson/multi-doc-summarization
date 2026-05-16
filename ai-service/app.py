import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import List

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from torch import nn
from transformers import PhobertTokenizer, RobertaConfig, RobertaModel


ROOT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT_DIR / "phobert_outputs" / "best_model.pt"
TOKENIZER_DIR = ROOT_DIR / "results" / "checkpoints_extractive" / "tokenizer"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CFG_DEFAULTS = {
  "MAX_LEN": 256,
  "DROPOUT": 0.30,
  "USE_NUMERIC_FEATURES": True,
  "NUMERIC_FEATURE_COLUMNS": (
    "sent_doc_pos_norm",
    "sent_clus_pos_norm",
    "n_words_norm",
    "title_overlap",
    "tag_overlap",
    "doc_size_norm",
    "topic_size_norm",
  ),
  "NUMERIC_FEATURE_PROJ_DIM": 16,
  "TASK_MODE": "auto",
  "USE_ADAPTIVE_BUDGET": True,
  "SUMMARY_MAX_SENTENCES": 12,
  "SUMMARY_MAX_WORDS": 450,
  "MIN_REQUIRED_SENTENCES": 3,
  "ADAPTIVE_MIN_SENTENCES": 3,
  "ADAPTIVE_SINGLE_MAX_SENTENCES": 8,
  "ADAPTIVE_MULTI_MAX_SENTENCES": 10,
  "ADAPTIVE_SINGLE_MAX_WORDS": 260,
  "ADAPTIVE_MULTI_MAX_WORDS": 320,
  "MMR_ALPHA": 0.65,
  "REDUNDANCY_WEIGHT": 0.30,
  "TOPIC_COVERAGE_WEIGHT": 0.15,
  "DOC_COVERAGE_WEIGHT": 0.06,
  "POSITION_BONUS_WEIGHT": 0.04,
  "TOPIC_RELEVANCE_WEIGHT": 0.12,
  "MAX_PER_TOPIC": 2,
  "MAX_PER_TOPIC_SINGLE": 7,
  "MAX_PER_TOPIC_MULTI": 2,
  "AUTO_MULTI_TOPIC_MIN_TOPICS": 3,
  "FILTER_WEAK_TOPICS_IN_MULTI": True,
  "MIN_TOPIC_SCORE_RATIO": 0.60,
  "MIN_SENT_SCORE": 0.20,
  "MAX_REDUNDANCY_JACCARD": 0.55,
  "FILTER_OFF_TOPIC_IN_SINGLE": True,
  "TOPIC_RELEVANCE_THRESHOLD": 0.06,
}

STOPWORDS = set("""
và là của có cho với trong trên dưới một những các được đã đang sẽ thì mà này đó
khi như về từ tại bởi vì do để hơn rất cũng không vào ra đến sau trước giữa
người việc năm ngày tháng theo nhiều ít lại nếu nên hay hoặc cùng mỗi
bên cạnh ngoài ra tuy nhiên dù vậy vẫn còn dù mặc dù
hiện nay gần đây tương lai quá trình thông qua nhằm đối với liên quan
có thể cần phải giúp khiến làm bị bởi đây kia ấy
""".split())
PHRASE_SPLIT = STOPWORDS | set("""
phát triển trở thành phổ biến nhanh chậm mạnh rõ rệt thường xuyên
hoạt động quá trình vấn đề điều yếu tố vai trò hình thức kết quả hiệu quả
cho phép phụ thuộc đòi hỏi gặp đối mặt sử dụng áp dụng triển khai
thông tin nội dung dịch vụ quyết định quan trọng hiện đại
""".split())
GENERIC = set("""
vấn đề hoạt động quá trình kết quả hiệu quả hình thức nền tảng hệ thống mô hình
dữ liệu thông tin dịch vụ nội dung phương pháp giải pháp yếu tố vai trò chất lượng
người dùng học sinh sinh viên doanh nghiệp chuyên gia bệnh viện
""".split())
BAD_PHRASES = {
  p.strip() for p in """
học tập lĩnh vực hình thức kết quả hiệu quả quốc gia mọi nơi
ngày tháng hiện nay gần đây tương lai vấn đề quan trọng hoạt động trực tuyến
""".split("\n") if p.strip()
}
BAD_EDGE = set("lĩnh vực hình thức kết quả hiệu quả quốc gia mọi nơi tập dễ xuất cận phép phụ thuộc".split())
DECODE_STOP = set("""
và là của có cho với trong trên dưới một những các được đã đang sẽ thì mà này đó
khi như về từ tại bởi vì do để hơn rất cũng không vào ra đến sau trước giữa
người việc năm ngày tháng theo nhiều ít lại nếu nên hay hoặc cùng mỗi
ra vào lên xuống làm bị bởi đây kia ấy nhằm thông qua
""".split())
for w in ["biến", "hiện", "phát"]:
  STOPWORDS.discard(w)
  PHRASE_SPLIT.discard(w)

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


class PhoBERTSentenceClassifier(nn.Module):
  def __init__(self, dropout: float, numeric_dim: int, proj_dim: int):
    super().__init__()
    self.encoder = RobertaModel(RobertaConfig(
      vocab_size=64001, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
      intermediate_size=3072, hidden_act="gelu", hidden_dropout_prob=0.1,
      attention_probs_dropout_prob=0.1, max_position_embeddings=258, type_vocab_size=1,
      pad_token_id=1, bos_token_id=0, eos_token_id=2, layer_norm_eps=1e-5,
    ))
    self.numeric_dim = int(numeric_dim or 0)
    if self.numeric_dim > 0:
      self.numeric_encoder = nn.Sequential(
        nn.LayerNorm(self.numeric_dim),
        nn.Linear(self.numeric_dim, max(1, proj_dim)),
        nn.GELU(),
        nn.Dropout(dropout * 0.5),
      )
      classifier_in = 768 + max(1, proj_dim)
    else:
      self.numeric_encoder = None
      classifier_in = 768
    self.dropout = nn.Dropout(dropout)
    self.classifier = nn.Sequential(
      nn.Linear(classifier_in, 384),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(384, 1),
    )

  def forward(self, input_ids, attention_mask=None, numeric_features=None):
    cls = self.dropout(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0])
    if self.numeric_encoder is not None:
      if numeric_features is None:
        numeric_features = torch.zeros((cls.size(0), self.numeric_dim), dtype=cls.dtype, device=cls.device)
      cls = torch.cat([cls, self.numeric_encoder(numeric_features.to(dtype=cls.dtype, device=cls.device))], dim=1)
    return self.classifier(cls).squeeze(-1)


def cfg_ns(raw: dict | None):
  merged = dict(CFG_DEFAULTS)
  if raw:
    merged.update(raw)
  return SimpleNamespace(**merged)


def normalize(text: str) -> str:
  if pd.isna(text):
    return ""
  text = unicodedata.normalize("NFKC", str(text)).replace("\xa0", " ")
  return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> List[str]:
  return re.findall(r"[\wÀ-ỹ]+", normalize(text).lower(), flags=re.UNICODE)


def count_words(text: str) -> int:
  return len(tokens(text))


def jaccard(a: str, b: str) -> float:
  ta, tb = set(tokens(a)), set(tokens(b))
  return 0.0 if not ta or not tb else len(ta & tb) / max(1, len(ta | tb))


def stable_hash(text: str) -> str:
  import hashlib
  return hashlib.md5(normalize(text).lower().encode("utf-8")).hexdigest()[:12]


def create_topic_ids_from_title_tags(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  df["topic_text"] = (df["title"].fillna("") + " " + df["tags"].fillna("")).map(normalize)
  df["topic_key"] = df["topic_text"].map(lambda value: stable_hash(value) if value else "")
  df["topic_id"] = -1
  for _, idxs in df.groupby("cluster_id", sort=False).groups.items():
    key_to_id = {}
    next_id = 0
    for idx in list(idxs):
      key = df.at[idx, "topic_key"] or f"DOC::{df.at[idx, 'doc_id']}"
      if key not in key_to_id:
        key_to_id[key] = next_id
        next_id += 1
      df.at[idx, "topic_id"] = key_to_id[key]
  return df


def model_input(row) -> str:
  return f"Tiêu đề: {normalize(row.get('title', ''))}. Từ khóa: {normalize(row.get('tags', ''))}. Câu: {normalize(row.get('sentence', ''))}"


def split_sentences(text: str) -> List[str]:
  text = normalize(text)
  if not text:
    return []
  for abbr in ["TP.", "TP.HCM.", "ThS.", "TS.", "PGS.", "GS.", "Ông.", "Bà.", "Mr.", "Ms.", "Dr.", "P.", "Q."]:
    text = text.replace(abbr, abbr.replace(".", "<DOT>"))
  chunks = [c.replace("<DOT>", ".").strip(" -*") for c in re.split(r"(?<=[.!?;:])\s+", re.sub(r"\s*\n+\s*", " ", text)) if c and c.strip()]
  if len(chunks) <= 1:
    chunks = [c.strip().replace("<DOT>", ".") for c in re.split(r",\s+", text) if c.strip()]
  return [c for c in chunks if count_words(c) >= 3]


def kw_norm(text: str) -> str:
  return re.sub(r"\s+", " ", re.sub(r"[^a-zA-ZÀ-ỹ0-9_\s]", " ", normalize(text).lower())).strip()


def kw_tokens(text: str):
  return re.findall(r"[a-zA-ZÀ-ỹ0-9_]+", kw_norm(text))


def kw_content(toks):
  return [t for t in toks if t not in STOPWORDS and len(t) >= 2]


def phrase_overlap(a: str, b: str) -> float:
  ta, tb = set(kw_norm(a).split()), set(kw_norm(b).split())
  return 0.0 if not ta or not tb else len(ta & tb) / max(1, min(len(ta), len(tb)))


def candidate_phrase(phrase: str) -> bool:
  toks = kw_norm(phrase).split()
  if not toks:
    return False
  content = kw_content(toks)
  p = " ".join(toks)
  if p in BAD_PHRASES or toks[0] in BAD_EDGE or toks[-1] in BAD_EDGE:
    return False
  if len(toks) == 1:
    return toks[0] not in STOPWORDS and toks[0] not in GENERIC and len(toks[0]) >= 3
  return toks[0] not in STOPWORDS and toks[-1] not in STOPWORDS and len(content) >= 2 and not all(t in GENERIC for t in content)


def redundant_phrase(phrase: str, selected: list) -> bool:
  p = kw_norm(phrase)
  pt = p.split()
  if not pt:
    return True
  for old in selected:
    o = kw_norm(old)
    ot = o.split()
    if f" {p} " in f" {o} " and len(pt) <= len(ot):
      return True
    if f" {o} " in f" {p} " and len(pt) <= len(ot) + 1:
      return True
    if phrase_overlap(p, o) >= 0.82:
      return True
  return False


def extract_keyword_phrases(content: str, top_k: int = 8, return_scores: bool = False):
  sents = split_sentences(content)
  if not sents:
    return []
  token_freq = Counter([t for t in kw_tokens(content) if t not in STOPWORDS and len(t) >= 2])
  max_tf = max(token_freq.values()) if token_freq else 1
  stats = defaultdict(lambda: {"freq": 0, "first_sent": 10**9, "first_pos": 10**9, "token_score": 0.0, "kind_bonus": 0.0})

  def chunks(sentence: str):
    toks, out, cur, start = kw_tokens(sentence), [], [], 0
    def flush(end_pos):
      nonlocal cur, start
      if cur:
        out.append((cur[:], start, end_pos))
        cur = []
    for i, t in enumerate(toks):
      if t in PHRASE_SPLIT or len(t) < 2 or t.isdigit():
        flush(i); start = i + 1
      else:
        if not cur:
          start = i
        cur.append(t)
    flush(len(toks))
    return out

  def subphrases(chunk_toks, start_pos):
    out, length = [], len(chunk_toks)
    if 2 <= length <= 5:
      p = " ".join(chunk_toks)
      if candidate_phrase(p):
        out.append((p, start_pos, "full"))
    if length > 5:
      for s, e, kind in [(0, 5, "head"), (max(0, length - 5), length, "tail")]:
        p = " ".join(chunk_toks[s:e])
        if candidate_phrase(p):
          out.append((p, start_pos + s, kind))
    for n in [4, 3, 2]:
      if length < n:
        continue
      for s in range(0, length - n + 1):
        is_edge = s == 0 or s + n == length
        if not is_edge and n <= 2:
          continue
        p = " ".join(chunk_toks[s:s + n])
        if candidate_phrase(p):
          out.append((p, start_pos + s, "edge_ngram" if is_edge else "inner_ngram"))
    return out

  for sent_idx, sent in enumerate(sents):
    for chunk_toks, start_pos, _ in chunks(sent):
      for phrase, pos, kind in subphrases(chunk_toks, start_pos):
        content_toks = kw_content(kw_norm(phrase).split())
        if len(content_toks) < 2:
          continue
        tok_score = sum(token_freq.get(t, 0) / max_tf for t in content_toks) / max(1, len(content_toks))
        st = stats[kw_norm(phrase)]
        st["freq"] += 1
        st["first_sent"] = min(st["first_sent"], sent_idx)
        st["first_pos"] = min(st["first_pos"], pos)
        st["token_score"] = max(st["token_score"], tok_score)
        st["kind_bonus"] = max(st["kind_bonus"], {"full": 0.60, "head": 0.24, "tail": 0.12, "edge_ngram": 0.06, "inner_ngram": 0.0}[kind])

  scored = []
  for phrase, st in stats.items():
    n = len(phrase.split())
    c = kw_content(phrase.split())
    score = (
      1.10 * math.log1p(st["freq"])
      + 0.82 * (1.0 / (1.0 + st["first_sent"]) + 0.18 / (1.0 + st["first_pos"]) + (0.45 if st["first_sent"] == 0 and st["first_pos"] == 0 else 0.0))
      + 0.52 * st["token_score"]
      + 0.32 * (len(c) / max(1, n))
      + {2: 0.15, 3: 0.30, 4: 0.42, 5: 0.30}.get(n, 0.0)
      + st["kind_bonus"]
      - (0.08 * (n - 4) if n >= 5 else 0.0)
    )
    scored.append((phrase, score))

  picked, out = [], []
  for phrase, score in sorted(scored, key=lambda x: (x[1], len(x[0].split())), reverse=True):
    if not candidate_phrase(phrase) or redundant_phrase(phrase, picked):
      continue
    picked.append(phrase)
    out.append((phrase, score))
    if len(picked) >= top_k:
      break
  return out if return_scores else picked


def auto_generate_tags(content: str) -> str:
  return ", ".join(extract_keyword_phrases(content, top_k=8, return_scores=False))


def auto_generate_title(content: str, max_words: int = 12) -> str:
  phrases = extract_keyword_phrases(content, top_k=10, return_scores=True)
  if not phrases:
    sents = split_sentences(content)
    if not sents:
      return ""
    words = sents[0].split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")
  main, main_score = phrases[0]
  aspect = None
  main_tokens = set(main.split())
  for p, s in phrases[1:]:
    p_tokens = set(p.split())
    if p_tokens and len(main_tokens | p_tokens) <= max_words and phrase_overlap(main, p) < 0.50 and s >= 0.45 * main_score:
      aspect = p
      break
  title = f"{main} và {aspect}" if aspect else main
  words = title.split()
  title = title if len(words) <= max_words else " ".join(words[:max_words])
  return title[:1].upper() + title[1:] if title else title


def add_numeric_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
  df = df.copy().sort_values(["cluster_id", "sent_clus_pos", "doc_id", "sent_doc_pos", "raw_index"], kind="stable").reset_index(drop=True)
  cluster_size = df.groupby("cluster_id")["sentence"].transform("size").astype(float)
  doc_size = df.groupby(["cluster_id", "doc_id"], dropna=False)["sentence"].transform("size").astype(float)
  topic_size = df.groupby(["cluster_id", "topic_id"], dropna=False)["sentence"].transform("size").astype(float)
  doc_order = df.groupby(["cluster_id", "doc_id"], dropna=False).cumcount().astype(float)
  cluster_order = df.groupby("cluster_id", dropna=False).cumcount().astype(float)
  sent_doc_pos = pd.to_numeric(df["sent_doc_pos"], errors="coerce").fillna(-1).astype(float)
  sent_clus_pos = pd.to_numeric(df["sent_clus_pos"], errors="coerce").fillna(-1).astype(float)
  safe_doc_pos = np.where(sent_doc_pos >= 0, sent_doc_pos, doc_order)
  safe_clus_pos = np.where(sent_clus_pos >= 0, sent_clus_pos, cluster_order)
  df["sent_doc_pos_norm"] = safe_doc_pos / np.maximum(doc_size.values - 1.0, 1.0)
  df["sent_clus_pos_norm"] = safe_clus_pos / np.maximum(cluster_size.values - 1.0, 1.0)
  n_words = pd.to_numeric(df["n_words"], errors="coerce").fillna(0).astype(float)
  df["n_words_norm"] = np.clip(np.log1p(n_words) / np.log1p(80.0), 0.0, 1.0)
  df["title_overlap"] = [jaccard(sent, title) for sent, title in zip(df["sentence"].fillna(""), df["title"].fillna(""))]
  df["tag_overlap"] = [jaccard(sent, tags) for sent, tags in zip(df["sentence"].fillna(""), df["tags"].fillna(""))]
  df["doc_size_norm"] = np.clip(doc_size.values / np.maximum(cluster_size.values, 1.0), 0.0, 1.0)
  df["topic_size_norm"] = np.clip(topic_size.values / np.maximum(cluster_size.values, 1.0), 0.0, 1.0)
  if getattr(cfg, "USE_NUMERIC_FEATURES", False):
    for col in cfg.NUMERIC_FEATURE_COLUMNS:
      df[col] = pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0).astype("float32").clip(lower=0.0, upper=1.0)
  return df


def build_custom_df(contents, cfg, cluster_id: str = "CUSTOM_CONTENT"):
  if isinstance(contents, str):
    contents = [contents]
  rows, global_pos = [], 0
  for doc_idx, content in enumerate(contents, start=1):
    title, tags = normalize(auto_generate_title(content)), normalize(auto_generate_tags(content))
    for sent_doc_pos, sentence in enumerate(split_sentences(normalize(content))):
      rows.append({
        "raw_index": global_pos,
        "cluster_id": cluster_id,
        "doc_id": doc_idx,
        "sent_clus_pos": global_pos,
        "sent_doc_pos": sent_doc_pos,
        "sentence": sentence,
        "title": title,
        "tags": tags,
        "oracle_score": 0.0,
        "n_words": count_words(sentence),
      })
      global_pos += 1
  df = pd.DataFrame(rows)
  if df.empty:
    raise ValueError("Không tách được câu nào.")
  df = add_numeric_features(create_topic_ids_from_title_tags(df), cfg)
  df["sent_global_index"] = np.arange(len(df))
  return df


def predict_scores(df: pd.DataFrame, model, tokenizer, cfg, batch_size: int = 32) -> np.ndarray:
  texts = [model_input(row) for _, row in df.iterrows()]
  feat_cols = list(cfg.NUMERIC_FEATURE_COLUMNS) if getattr(cfg, "USE_NUMERIC_FEATURES", False) else []
  numeric = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32").values if feat_cols else np.zeros((len(df), 0), dtype="float32")
  probs = []
  with torch.no_grad():
    for start in range(0, len(texts), batch_size):
      encoded = tokenizer(texts[start:start + batch_size], truncation=True, max_length=int(cfg.MAX_LEN), padding=True, return_attention_mask=True, return_token_type_ids=False, return_tensors="pt")
      encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
      if feat_cols:
        encoded["numeric_features"] = torch.tensor(numeric[start:start + batch_size], dtype=torch.float32, device=DEVICE)
      probs.extend(torch.sigmoid(model(**encoded)).detach().cpu().numpy().tolist())
  return np.asarray(probs, dtype=float)


def add_scores(df: pd.DataFrame, probs: np.ndarray, cfg) -> pd.DataFrame:
  out = df.reset_index(drop=True).copy()
  out["_orig_idx"] = np.arange(len(out))
  out["model_score"] = probs
  out["position_score"] = out.apply(lambda row: 0.7 * (1 / (1 + max(0, row.get("sent_doc_pos", 999)))) + 0.3 * (1 / math.sqrt(1 + max(0, row.get("sent_clus_pos", 999)))), axis=1)
  freq = Counter([t for t in tokens(" ".join(out["sentence"].fillna("").astype(str).tolist())) if len(t) >= 2 and t not in DECODE_STOP and not t.isdigit()])
  repeated = {t for t, c in freq.items() if c >= 2} or {t for t, _ in freq.most_common(20)}
  def relevance(sent: str) -> float:
    toks = {t for t in tokens(sent) if len(t) >= 2 and t not in DECODE_STOP and not t.isdigit()}
    if not toks or not repeated:
      return 0.0
    return max(len(toks & repeated) / max(1, len(toks)), float(np.mean([jaccard(sent, other) for other in out["sentence"].tolist() if other != sent])) if len(out) > 1 else 0.0)
  out["topic_relevance"] = out["sentence"].apply(relevance)
  out["final_score"] = cfg.MMR_ALPHA * out["model_score"] + cfg.POSITION_BONUS_WEIGHT * out["position_score"] + cfg.TOPIC_RELEVANCE_WEIGHT * out["topic_relevance"]
  return out


def clamp_value(value: int, low: int, high: int) -> int:
  return max(low, min(high, value))


def runtime_cfg(cfg, scored: pd.DataFrame, mode: str):
  n_sents = len(scored)
  if n_sents <= 8:
    sent_ratio = 0.65
  elif n_sents <= 16:
    sent_ratio = 0.52
  elif n_sents <= 30:
    sent_ratio = 0.38
  else:
    sent_ratio = 0.28
  target_sents = int(round(n_sents * sent_ratio))
  target_words = int(scored["sentence"].map(count_words).sum() * (0.42 if mode == "multi_topic_coverage" else 0.45))
  out = SimpleNamespace(**vars(cfg))
  out.SUMMARY_MAX_SENTENCES = min(n_sents, clamp_value(target_sents, 3, cfg.ADAPTIVE_MULTI_MAX_SENTENCES if mode == "multi_topic_coverage" else cfg.ADAPTIVE_SINGLE_MAX_SENTENCES))
  out.SUMMARY_MAX_WORDS = clamp_value(target_words, 80, cfg.ADAPTIVE_MULTI_MAX_WORDS if mode == "multi_topic_coverage" else cfg.ADAPTIVE_SINGLE_MAX_WORDS)
  out.MIN_REQUIRED_SENTENCES = min(3, out.SUMMARY_MAX_SENTENCES)
  out.RUNTIME_BUDGET = {"target_sents": out.SUMMARY_MAX_SENTENCES, "target_words": out.SUMMARY_MAX_WORDS, "min_required": out.MIN_REQUIRED_SENTENCES}
  return out


def select_sentences(scored: pd.DataFrame, cfg):
  mode = "multi_topic_coverage" if (cfg.TASK_MODE == "multi_topic_coverage" or (cfg.TASK_MODE == "auto" and scored["topic_id"].nunique() >= cfg.AUTO_MULTI_TOPIC_MIN_TOPICS)) else "single_topic_enrichment"
  rcfg = runtime_cfg(cfg, scored, mode)
  selected, selected_texts, selected_docs, topic_counts, total_words = [], [], set(), Counter(), 0
  base = scored if mode == "multi_topic_coverage" else (scored[scored["topic_relevance"] >= cfg.TOPIC_RELEVANCE_THRESHOLD] if len(scored) > cfg.MIN_REQUIRED_SENTENCES else scored)
  valid_topics = base.groupby("topic_id")["final_score"].mean().sort_values(ascending=False).index.tolist()
  pool = valid_topics if mode == "multi_topic_coverage" else [None]
  while len(selected) < rcfg.SUMMARY_MAX_SENTENCES:
    added = False
    for topic_id in pool:
      best, best_score = None, -1e18
      rows = base if topic_id is None else base[base["topic_id"] == topic_id]
      for _, row in rows.sort_values("final_score", ascending=False).iterrows():
        if int(row["_orig_idx"]) in selected:
          continue
        if row["model_score"] < cfg.MIN_SENT_SCORE and len(selected) >= rcfg.MIN_REQUIRED_SENTENCES:
          continue
        words = count_words(row["sentence"])
        if total_words + words > rcfg.SUMMARY_MAX_WORDS and selected:
          continue
        redundancy = max([jaccard(row["sentence"], text) for text in selected_texts], default=0.0)
        if selected and redundancy > cfg.MAX_REDUNDANCY_JACCARD:
          continue
        doc_bonus = 1.0 if row["doc_id"] not in selected_docs else 0.0
        topic_bonus = 1.0 if topic_id is not None and topic_counts[topic_id] == 0 else 0.0
        score = float(row["final_score"]) - cfg.REDUNDANCY_WEIGHT * redundancy + cfg.DOC_COVERAGE_WEIGHT * doc_bonus + (cfg.TOPIC_COVERAGE_WEIGHT * topic_bonus if topic_id is not None else 0.0)
        if score > best_score:
          best, best_score = row, score
      if best is not None:
        selected.append(int(best["_orig_idx"]))
        selected_texts.append(best["sentence"])
        selected_docs.add(best["doc_id"])
        if topic_id is not None:
          topic_counts[topic_id] += 1
        total_words += count_words(best["sentence"])
        added = True
        if mode == "single_topic_enrichment":
          break
    if not added:
      break
  if not selected and len(scored):
    selected = [int(scored.sort_values("final_score", ascending=False).iloc[0]["_orig_idx"])]
  cfg.LAST_RUNTIME_BUDGET = rcfg.RUNTIME_BUDGET
  return sorted(selected, key=lambda idx: (scored.set_index("_orig_idx").loc[idx]["sent_clus_pos"], scored.set_index("_orig_idx").loc[idx]["doc_id"], scored.set_index("_orig_idx").loc[idx]["sent_doc_pos"])), mode


def summarize_documents(documents: List[str], model, tokenizer, cfg):
  df = build_custom_df(documents, cfg)
  scored = add_scores(df, predict_scores(df, model, tokenizer, cfg), cfg)
  selected, mode = select_sentences(scored, cfg)
  selected_info = scored.set_index("_orig_idx").loc[selected].reset_index()
  return {
    "summary": " ".join(df.iloc[idx]["sentence"] for idx in selected),
    "selected_info": selected_info,
    "selected_indices": selected,
    "scored_df": scored,
    "decoding_mode": mode,
    "runtime_budget": getattr(cfg, "LAST_RUNTIME_BUDGET", None),
  }


@lru_cache(maxsize=1)
def load_artifacts():
  checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
  cfg = cfg_ns(checkpoint.get("cfg"))
  model = PhoBERTSentenceClassifier(float(cfg.DROPOUT), len(cfg.NUMERIC_FEATURE_COLUMNS) if cfg.USE_NUMERIC_FEATURES else 0, int(cfg.NUMERIC_FEATURE_PROJ_DIM))
  model.load_state_dict(checkpoint["model_state_dict"])
  model.to(DEVICE)
  model.eval()
  tokenizer = PhobertTokenizer.from_pretrained(str(TOKENIZER_DIR))
  return model, tokenizer, cfg, checkpoint


@app.on_event("startup")
def startup_event():
  load_artifacts()


@app.get("/health")
def health():
  _, _, _, checkpoint = load_artifacts()
  return {"status": "ok", "device": DEVICE, "model_path": str(CHECKPOINT_PATH), "threshold": checkpoint.get("threshold"), "best_epoch": checkpoint.get("best_epoch"), "best_score": checkpoint.get("best_score")}


@app.post("/api/summarize", response_model=SummarizeResponse)
def summarize(payload: SummarizeRequest):
  documents = [d.strip() for d in payload.documents if d and d.strip()]
  if not documents:
    return SummarizeResponse(summary="", sentences=[], meta={"message": "No documents provided"})
  model, tokenizer, cfg, checkpoint = load_artifacts()
  result = summarize_documents(documents, model, tokenizer, cfg)
  return SummarizeResponse(
    summary=result["summary"],
    sentences=[
      SentenceCandidate(text=row["sentence"], document_index=int(row["doc_id"]) - 1, probability=round(float(row["model_score"]), 4))
      for _, row in result["selected_info"].iterrows()
    ],
    meta={
      "device": DEVICE,
      "candidate_count": int(len(result["scored_df"])),
      "selected_count": int(len(result["selected_indices"])),
      "threshold": checkpoint.get("threshold"),
      "decoding_mode": result["decoding_mode"],
      "runtime_budget": result["runtime_budget"],
    },
  )
