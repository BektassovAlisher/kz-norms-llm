"""Пути проекта в одном месте."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
SYNTHETIC = DATA / "synthetic"
SFT = DATA / "sft"
EVAL = DATA / "eval"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

SOURCES_CSV = DATA / "sources.csv"
CLAUSES_JSONL = PROCESSED / "clauses.jsonl"
INDEX = ARTIFACTS / "index"  # chunks.jsonl + BM25
CHUNKS_JSONL = INDEX / "chunks.jsonl"
BM25_DIR = INDEX / "bm25"
