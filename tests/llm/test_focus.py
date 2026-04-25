from llms.intent.focus import rewrite_known_term_aliases


def test_rewrite_known_term_aliases_uses_asset_backed_rules():
    assert rewrite_known_term_aliases("康夫ui 教程") == "康夫ui 教程"
    assert rewrite_known_term_aliases("康夫 UI 教程") == "康夫 UI 教程"


def test_rewrite_known_term_aliases_keeps_unknown_terms_unchanged():
    assert rewrite_known_term_aliases("黑神话 悟空") == "黑神话 悟空"
