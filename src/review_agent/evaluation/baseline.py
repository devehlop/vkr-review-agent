"""Словарный метод без LLM — нижняя граница качества для сравнения в главе 3.

Атрибут находится по основам слов, тональность — по словарю оценочных
основ с учётом частицы «не» перед словом. Метод намеренно простой: он
показывает, сколько даёт LLM по сравнению с правилами.
"""

from __future__ import annotations

import re

# Код атрибута → основы слов (сравнение по началу слова)
ATTRIBUTE_STEMS: dict[str, list[str]] = {
    "size": ["размер", "маломер", "большемер", "мал", "велик", "впритык", "сел", "сидит", "длин", "коротк", "рукав", "рост", "сетк", "посадк", "обтяг"],
    "fabric": ["ткан", "материал", "синтет", "хлоп", "тонк", "плотн", "мягк", "колюч", "просвеч", "состав", "трикотаж", "шерст", "флис"],
    "sewing": ["шв", "шов", "нитк", "нит", "пошив", "строчк", "дыр", "брак", "молни", "пуговиц", "запах", "воняет", "криво", "пятн"],
    "look": ["цвет", "рисунок", "принт", "фасон", "красив", "смотрит", "выгляд", "ярк", "блекл"],
    "match": ["фото", "картинк", "описани", "соответств"],
    "durability": ["стирк", "постира", "линя", "катышк", "полинял", "растянул"],
    "price": ["цен", "деньг", "дорог", "дешев", "стоит", "руб"],
    "delivery": ["доставк", "пришл", "пришел", "пришёл", "шел", "шёл", "дошл", "трек", "посылк", "почт", "недел", "месяц"],
    "packaging": ["упаков", "пакет", "коробк"],
    "seller": ["продав", "спор", "магазин", "возврат", "вернул", "прислал"],
    "overall": ["рекоменд", "совет", "доволь", "понрав", "ужас", "отлич", "супер", "класс"],
}

POSITIVE = ["хорош", "отлич", "красив", "мягк", "приятн", "довол", "понрав", "рекоменд", "супер", "класс", "подош", "соответств", "быстр", "спасиб", "качествен", "идеальн", "норм"]
NEGATIVE = ["плох", "ужас", "кошмар", "отврат", "тонк", "синтет", "крив", "дыр", "брак", "маломер", "большемер", "мал", "велик", "долг", "разочар", "торч", "линя", "катышк", "воня", "дешев", "блекл", "просвеч", "колюч", "обман", "жаль", "зря"]

_WORD = re.compile(r"[а-яё]+", re.IGNORECASE)


def _words(text: str) -> list[str]:
    return [w.lower().replace("ё", "е") for w in _WORD.findall(text)]


def _starts(word: str, stems: list[str]) -> bool:
    return any(word.startswith(s.replace("ё", "е")) for s in stems)


def sentence_sentiment(text: str) -> int:
    score = 0
    words = _words(text)
    for i, w in enumerate(words):
        sign = -1 if i > 0 and words[i - 1] == "не" else 1
        if _starts(w, POSITIVE):
            score += sign
        elif _starts(w, NEGATIVE):
            score -= sign
    return (score > 0) - (score < 0)


def predict(sentences: dict[str, str]) -> dict[str, dict[str, int]]:
    """id → {код атрибута: тональность}. Тональность одна на всё предложение."""
    result: dict[str, dict[str, int]] = {}
    for sid, text in sentences.items():
        words = _words(text)
        attrs = [code for code, stems in ATTRIBUTE_STEMS.items() if any(_starts(w, stems) for w in words)]
        # «Общее впечатление» только если нет конкретики
        if len(attrs) > 1 and "overall" in attrs:
            attrs.remove("overall")
        sentiment = sentence_sentiment(text)
        result[sid] = {code: sentiment for code in attrs}
    return result
