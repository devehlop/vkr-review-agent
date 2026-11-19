"""Хранилище SQLite (docs/architecture.md, раздел 6).

Один файл базы на все запуски. Каждый запуск анализа — строка в run,
все остальные таблицы ссылаются на неё. Таблица llm_call заполняется
из LLM-шлюза через on_record и нужна для NFR-3 и NFR-5.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from review_agent.llm.gateway import LLMCallRecord
from review_agent.models import Issue, Mention, Product, Recommendation, Review, Sentence

SCHEMA = """
CREATE TABLE IF NOT EXISTS run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    taxonomy TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'running',
    params TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS product (
    run_id INTEGER NOT NULL REFERENCES run(id),
    id TEXT NOT NULL,
    name TEXT NOT NULL,
    is_own INTEGER NOT NULL,
    marketplace TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, id)
);
CREATE TABLE IF NOT EXISTS review (
    run_id INTEGER NOT NULL REFERENCES run(id),
    id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    text TEXT NOT NULL,
    pros TEXT NOT NULL,
    cons TEXT NOT NULL,
    rating INTEGER,
    created TEXT,
    variant TEXT NOT NULL DEFAULT '',
    suspicious INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, id)
);
CREATE TABLE IF NOT EXISTS sentence (
    run_id INTEGER NOT NULL REFERENCES run(id),
    id TEXT NOT NULL,
    review_id TEXT NOT NULL,
    text TEXT NOT NULL,
    source_field TEXT NOT NULL,
    PRIMARY KEY (run_id, id)
);
CREATE TABLE IF NOT EXISTS mention (
    run_id INTEGER NOT NULL REFERENCES run(id),
    id TEXT NOT NULL,
    sentence_id TEXT NOT NULL,
    attribute TEXT NOT NULL,
    feature TEXT NOT NULL,
    sentiment INTEGER NOT NULL CHECK (sentiment IN (-1, 0, 1)),
    quote TEXT NOT NULL,
    PRIMARY KEY (run_id, id)
);
CREATE TABLE IF NOT EXISTS issue (
    run_id INTEGER NOT NULL REFERENCES run(id),
    id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    attribute TEXT NOT NULL,
    title TEXT NOT NULL,
    coverage REAL NOT NULL,
    neg_share REAL NOT NULL,
    mention_ids TEXT NOT NULL,
    PRIMARY KEY (run_id, id)
);
CREATE TABLE IF NOT EXISTS recommendation (
    run_id INTEGER NOT NULL REFERENCES run(id),
    id TEXT NOT NULL,
    issue_id TEXT,
    action TEXT NOT NULL,
    type TEXT NOT NULL,
    rationale TEXT NOT NULL,
    expected_effect TEXT NOT NULL,
    effort INTEGER NOT NULL,
    quotes TEXT NOT NULL,
    priority REAL,
    grounded INTEGER,
    PRIMARY KEY (run_id, id)
);
CREATE TABLE IF NOT EXISTS critique (
    run_id INTEGER NOT NULL REFERENCES run(id),
    recommendation_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    scores TEXT NOT NULL,
    comment TEXT NOT NULL,
    passed INTEGER NOT NULL,
    PRIMARY KEY (run_id, recommendation_id, iteration)
);
CREATE TABLE IF NOT EXISTS llm_call (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES run(id),
    created_at TEXT NOT NULL,
    step TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    prompt_hash TEXT NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    cost REAL NOT NULL,
    latency REAL NOT NULL,
    cached INTEGER NOT NULL,
    ok INTEGER NOT NULL,
    error TEXT NOT NULL,
    response TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Storage:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # --- запуск ---------------------------------------------------------

    def start_run(self, provider: str, model: str, prompt_version: str, taxonomy: str = "", params: dict | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO run (started_at, provider, model, prompt_version, taxonomy, params) VALUES (?, ?, ?, ?, ?, ?)",
            (_now(), provider, model, prompt_version, taxonomy, json.dumps(params or {}, ensure_ascii=False)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str = "done") -> None:
        self.conn.execute("UPDATE run SET finished_at = ?, status = ? WHERE id = ?", (_now(), status, run_id))
        self.conn.commit()

    # --- сохранение -----------------------------------------------------

    def save_products(self, run_id: int, products: list[Product]) -> None:
        self.conn.executemany(
            "INSERT INTO product VALUES (?, ?, ?, ?, ?)",
            [(run_id, p.id, p.name, int(p.is_own), p.marketplace) for p in products],
        )
        self.conn.commit()

    def save_reviews(self, run_id: int, reviews: list[Review]) -> None:
        self.conn.executemany(
            "INSERT INTO review VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (run_id, r.id, r.product_id, r.text, r.pros, r.cons, r.rating,
                 r.created.isoformat() if r.created else None, r.variant, int(r.suspicious))
                for r in reviews
            ],
        )
        self.conn.commit()

    def save_sentences(self, run_id: int, sentences: list[Sentence]) -> None:
        self.conn.executemany(
            "INSERT INTO sentence VALUES (?, ?, ?, ?, ?)",
            [(run_id, s.id, s.review_id, s.text, s.source_field) for s in sentences],
        )
        self.conn.commit()

    def save_mentions(self, run_id: int, mentions: list[Mention]) -> None:
        self.conn.executemany(
            "INSERT INTO mention VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(run_id, m.id, m.sentence_id, m.attribute, m.feature, m.sentiment, m.quote) for m in mentions],
        )
        self.conn.commit()

    def save_issues(self, run_id: int, issues: list[Issue]) -> None:
        self.conn.executemany(
            "INSERT INTO issue VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(run_id, i.id, i.product_id, i.attribute, i.title, i.coverage, i.neg_share, json.dumps(i.mention_ids)) for i in issues],
        )
        self.conn.commit()

    def save_recommendations(self, run_id: int, recs: list[Recommendation]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO recommendation VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (run_id, r.id, r.issue_id, r.action, r.type.value, r.rationale, r.expected_effect, r.effort,
                 json.dumps(r.quotes, ensure_ascii=False), r.priority, None if r.grounded is None else int(r.grounded))
                for r in recs
            ],
        )
        self.conn.commit()

    def save_critique(self, run_id: int, recommendation_id: str, iteration: int, scores: dict, comment: str, passed: bool) -> None:
        self.conn.execute(
            "INSERT INTO critique VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, recommendation_id, iteration, json.dumps(scores, ensure_ascii=False), comment, int(passed)),
        )
        self.conn.commit()

    def llm_logger(self, run_id: int | None):
        """Функция для LLMGateway(on_record=...): пишет каждый запрос в llm_call."""

        def log(r: LLMCallRecord) -> None:
            self.conn.execute(
                "INSERT INTO llm_call (run_id, created_at, step, provider, model, attempt, prompt_hash, tokens_in, "
                "tokens_out, cost, latency, cached, ok, error, response) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, _now(), r.step, r.provider, r.model, r.attempt, r.prompt_hash, r.tokens_in, r.tokens_out,
                 r.cost, r.latency, int(r.cached), int(r.ok), r.error, r.response),
            )
            self.conn.commit()

        return log

    # --- чтение ---------------------------------------------------------

    def load_mentions(self, run_id: int) -> list[Mention]:
        rows = self.conn.execute("SELECT * FROM mention WHERE run_id = ? ORDER BY rowid", (run_id,)).fetchall()
        return [Mention(id=r["id"], sentence_id=r["sentence_id"], attribute=r["attribute"], feature=r["feature"],
                        sentiment=r["sentiment"], quote=r["quote"]) for r in rows]

    def load_sentences(self, run_id: int) -> list[Sentence]:
        rows = self.conn.execute("SELECT * FROM sentence WHERE run_id = ? ORDER BY rowid", (run_id,)).fetchall()
        return [Sentence(id=r["id"], review_id=r["review_id"], text=r["text"], source_field=r["source_field"]) for r in rows]

    def run_cost(self, run_id: int) -> dict[str, float]:
        """Итог по запуску: запросы, токены, стоимость, время (NFR-3)."""
        r = self.conn.execute(
            "SELECT SUM(1 - cached) AS calls, SUM(tokens_in) AS tokens_in, SUM(tokens_out) AS tokens_out, "
            "SUM(cost) AS cost, SUM(latency) AS latency, SUM(1 - ok) AS errors FROM llm_call WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return {k: (r[k] or 0) for k in r.keys()}
