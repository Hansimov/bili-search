from llms.intent.focus import extract_focus_spans
from llms.intent.focus import rewrite_known_term_aliases


def test_rewrite_known_term_aliases_uses_asset_backed_rules():
    assert rewrite_known_term_aliases("康夫ui 教程") == "ComfyUI 教程"
    assert rewrite_known_term_aliases("康夫 UI 教程") == "ComfyUI 教程"


def test_rewrite_known_term_aliases_keeps_unknown_terms_unchanged():
    assert rewrite_known_term_aliases("黑神话 悟空") == "黑神话 悟空"


def test_extract_focus_spans_prefers_quoted_owner_name():
    assert extract_focus_spans('"这里是小天啊" 是一个 UP主名字') == [
        "这里是小天啊"
    ]
