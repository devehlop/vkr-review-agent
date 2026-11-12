"""Хранилище SQLite: отзывы, предложения, упоминания, проблемы, рекомендации.

Промежуточные результаты сохраняются, чтобы шаг конвейера можно было перезапустить
без повтора предыдущих (и без повторной оплаты запросов к модели).
Журнал запросов к LLM ведётся отдельно шлюзом в файле JSONL.
"""
import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

from review_agent.models import Critique, Issue, Recommendation, Review, Sentence
from review_agent.pipeline.extract import Mention

SCHEMA = """
CREATE TABLE IF NOT EXISTS run (
    id INTEGER PRIMARY KEY, started_at TEXT, model TEXT, prompt_version TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS review (
    run_id INTEGER, id INTEGER, product TEXT, is_own INTEGER, text TEXT, pros TEXT, cons TEXT,
    rating INTEGER, created TEXT, variant TEXT, PRIMARY KEY (run_id, id));
CREATE TABLE IF NOT EXISTS sentence (
    run_id INTEGER, id INTEGER, review_id INTEGER, text TEXT, source_field TEXT, PRIMARY KEY (run_id, id));
CREATE TABLE IF NOT EXISTS mention (
    run_id INTEGER, sentence_id INTEGER, attribute TEXT, feature TEXT, sentiment INTEGER, quote TEXT);
CREATE TABLE IF NOT EXISTS issue (
    run_id INTEGER, id INTEGER, product TEXT, attribute TEXT, title TEXT, sentence_ids TEXT,
    coverage REAL, neg_share REAL, rarer_at_competitors INTEGER, PRIMARY KEY (run_id, id));
CREATE TABLE IF NOT EXISTS recommendation (
    run_id INTEGER, id INTEGER, issue_id INTEGER, action TEXT, type TEXT, rationale TEXT, effort INTEGER,
    quotes TEXT, priority REAL, grounded INTEGER, expected_effect TEXT, PRIMARY KEY (run_id, id));
CREATE TABLE IF NOT EXISTS critique (
    run_id INTEGER, recommendation_id INTEGER, iteration INTEGER, scores TEXT, comment TEXT, passed INTEGER);
CREATE INDEX IF NOT EXISTS mention_run ON mention (run_id, sentence_id);
"""


class Storage:
    def __init__(self, path: str | Path = "review_agent.db"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    # --- запуск анализа ---
    def start_run(self, model: str, prompt_version: str) -> int:
        cursor = self._db.execute(
            "INSERT INTO run (started_at, model, prompt_version, status) VALUES (?, ?, ?, 'идёт')",
            (datetime.now().isoformat(timespec="seconds"), model, prompt_version))
        self._db.commit()
        return cursor.lastrowid

    def finish_run(self, run_id: int, status: str = "готово") -> None:
        self._db.execute("UPDATE run SET status = ? WHERE id = ?", (status, run_id))
        self._db.commit()

    def run_status(self, run_id: int) -> str | None:
        row = self._db.execute("SELECT status FROM run WHERE id = ?", (run_id,)).fetchone()
        return row[0] if row else None

    def _replace(self, table: str, run_id: int, rows: list[tuple]) -> None:
        """Перезаписывает результат шага: повторный запуск шага не создаёт дублей."""
        self._db.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
        if rows:
            marks = ", ".join("?" * (len(rows[0]) + 1))
            self._db.executemany(f"INSERT INTO {table} VALUES ({marks})", [(run_id, *row) for row in rows])
        self._db.commit()

    # --- отзывы и предложения ---
    def save_reviews(self, run_id: int, reviews: list[Review]) -> None:
        self._replace("review", run_id, [
            (r.id, r.product, int(r.is_own), r.text, r.pros, r.cons, r.rating,
             r.created.isoformat() if r.created else None, r.variant) for r in reviews])

    def load_reviews(self, run_id: int) -> list[Review]:
        rows = self._db.execute(
            "SELECT id, product, is_own, text, pros, cons, rating, created, variant FROM review "
            "WHERE run_id = ? ORDER BY id", (run_id,))
        return [Review(id=i, product=p, is_own=bool(own), text=t, pros=pr, cons=c, rating=rt,
                       created=date.fromisoformat(cr) if cr else None, variant=v)
                for i, p, own, t, pr, c, rt, cr, v in rows]

    def save_sentences(self, run_id: int, sentences: list[Sentence]) -> None:
        self._replace("sentence", run_id, [(s.id, s.review_id, s.text, s.source_field) for s in sentences])

    def load_sentences(self, run_id: int) -> list[Sentence]:
        rows = self._db.execute(
            "SELECT id, review_id, text, source_field FROM sentence WHERE run_id = ? ORDER BY id", (run_id,))
        return [Sentence(*row) for row in rows]

    # --- упоминания (результат шага 4) ---
    def save_mentions(self, run_id: int, mentions: dict[int, list[Mention]]) -> None:
        self._replace("mention", run_id, [
            (sentence_id, m.attribute, m.feature, m.sentiment, m.quote)
            for sentence_id, items in mentions.items() for m in items])

    def load_mentions(self, run_id: int) -> dict[int, list[Mention]]:
        result: dict[int, list[Mention]] = {}
        rows = self._db.execute(
            "SELECT sentence_id, attribute, feature, sentiment, quote FROM mention WHERE run_id = ? ORDER BY rowid",
            (run_id,))
        for sentence_id, attribute, feature, sentiment, quote in rows:
            result.setdefault(sentence_id, []).append(
                Mention(attribute=attribute, feature=feature, sentiment=sentiment, quote=quote))
        return result

    # --- проблемы и рекомендации ---
    def save_issues(self, run_id: int, issues: list[Issue]) -> None:
        self._replace("issue", run_id, [
            (i.id, i.product, i.attribute, i.title, json.dumps(i.sentence_ids), i.coverage, i.neg_share,
             int(i.rarer_at_competitors)) for i in issues])

    def load_issues(self, run_id: int) -> list[Issue]:
        rows = self._db.execute(
            "SELECT id, product, attribute, title, sentence_ids, coverage, neg_share, rarer_at_competitors "
            "FROM issue WHERE run_id = ? ORDER BY id", (run_id,))
        return [Issue(i, p, a, t, json.loads(ids), cov, neg, bool(rarer)) for i, p, a, t, ids, cov, neg, rarer in rows]

    def save_recommendations(self, run_id: int, items: list[Recommendation]) -> None:
        self._replace("recommendation", run_id, [
            (r.id, r.issue_id, r.action, r.type, r.rationale, r.effort,
             json.dumps(r.quotes, ensure_ascii=False), r.priority, int(r.grounded), r.expected_effect) for r in items])

    def load_recommendations(self, run_id: int) -> list[Recommendation]:
        rows = self._db.execute(
            "SELECT id, issue_id, action, type, rationale, effort, quotes, priority, grounded, expected_effect "
            "FROM recommendation WHERE run_id = ? ORDER BY priority DESC, id", (run_id,))
        return [Recommendation(i, iss, a, t, r, e, [tuple(q) for q in json.loads(q_)], p, bool(g), eff or "")
                for i, iss, a, t, r, e, q_, p, g, eff in rows]

    def save_critiques(self, run_id: int, items: list[Critique]) -> None:
        """Все проверки критика, включая отклонённые рекомендации: материал для оценки вклада критика."""
        self._replace("critique", run_id, [
            (c.recommendation_id, c.iteration, json.dumps(c.scores), c.comment, int(c.passed)) for c in items])

    def load_critiques(self, run_id: int) -> list[Critique]:
        rows = self._db.execute(
            "SELECT recommendation_id, iteration, scores, comment, passed FROM critique WHERE run_id = ? ORDER BY rowid",
            (run_id,))
        return [Critique(r, i, json.loads(s), c, bool(p)) for r, i, s, c, p in rows]
