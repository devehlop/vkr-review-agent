from review_agent.models import Review
from review_agent.preprocess.pii import find_pii, has_pii, mask_pii
from review_agent.preprocess.text import (
    clean_text,
    deduplicate,
    prepare_review,
    review_sentences,
    split_sentences,
)


def test_mask_phone_email_url_order():
    text = (
        "Пишите syringal@yandex.ru или звоните 89198553809, "
        "+7 (912) 345-67-89. Фото тут: http://s.aliexpress.com/q2IFfamU "
        "заказ 84811276780423"
    )
    masked = mask_pii(text)
    assert "yandex.ru" not in masked
    assert "8919" not in masked and "345-67-89" not in masked
    assert "aliexpress" not in masked
    assert "84811276780423" not in masked
    assert masked.count("[ТЕЛЕФОН]") == 2
    assert {"[EMAIL]", "[ССЫЛКА]", "[НОМЕР]"} <= set(find_pii(text))


def test_sizes_and_prices_are_not_pii():
    text = "Размер S на 92-60-90, рост 170, цена 1500 руб, заказывала 15.06.2023"
    assert not has_pii(text)
    assert mask_pii(text) == text


def test_clean_text():
    assert clean_text("  Отлично!!!!!!!  \r\n\r\n\r\nВ размер  ") == "Отлично!!!\nВ размер"
    assert clean_text(None) == ""


def test_split_sentences_by_punct_and_newlines():
    text = "Ткань тонкая. Швы кривые!\nРазмер подошел\n)))"
    assert split_sentences(text) == ["Ткань тонкая.", "Швы кривые!", "Размер подошел"]


def test_review_sentences_keep_source_field():
    r = Review(id="r1", product_id="p", pros="Мягкая ткань", cons="Маломерит. Нитки торчат", text="В целом норм")
    sents = review_sentences(r)
    assert [(s.source_field, s.text) for s in sents] == [
        ("pros", "Мягкая ткань"),
        ("cons", "Маломерит."),
        ("cons", "Нитки торчат"),
        ("comment", "В целом норм"),
    ]
    assert len({s.id for s in sents}) == 4


def test_prepare_review_masks_pii():
    r = prepare_review(Review(id="r", product_id="p", text="Продам, пишите a.b@mail.ru"))
    assert "mail.ru" not in r.text


def test_deduplicate():
    reviews = [
        Review(id="1", product_id="p", text="Отличное платье!"),
        Review(id="2", product_id="p", text="отличное  платье"),
        Review(id="3", product_id="q", text="Отличное платье!"),
        Review(id="4", product_id="p", text="  !!! "),
    ]
    kept, removed = deduplicate(reviews)
    assert [r.id for r in kept] == ["1", "3"]
    assert removed == 2
