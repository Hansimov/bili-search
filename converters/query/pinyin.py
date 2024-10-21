import pypinyin
import zhconv

from itertools import product
from pypinyin.constants import RE_HANS


class ChinesePinyinizer:
    def text_to_simple(self, text: str) -> str:
        return zhconv.convert(text, "zh-cn")

    def text_to_segs(self, text: str) -> list[str]:
        segs = []
        for ch in text:
            if RE_HANS.match(ch):
                segs.append(ch)
            else:
                if segs and not RE_HANS.match(segs[-1]):
                    segs[-1] += ch
                else:
                    segs.append(ch)
        return segs

    def text_to_pinyin_choices(self, text: str) -> list[list[str]]:
        return pypinyin.pinyin(text, style=pypinyin.STYLE_NORMAL, heteronym=True)

    def text_to_pinyin_combinations(self, text: str) -> list[list[str]]:
        text = self.text_to_simple(text)
        pinyin_choices = self.text_to_pinyin_choices(text)
        pinyin_combinations = list(product(*pinyin_choices))
        return pinyin_combinations

    def text_to_pinyin_segs(self, text: str) -> list[str]:
        return self.text_to_pinyin_combinations(text)[0]

    def pinyin_choices_to_str(
        self, pinyin_choices: list[list[str]], sep: str = ""
    ) -> str:
        return sep.join([choice[0] for choice in pinyin_choices])

    def text_to_pinyin_str(self, text: str, sep: str = "") -> str:
        text = self.text_to_simple(text)
        pinyin_choices = self.text_to_pinyin_choices(text)
        pinyin_str = self.pinyin_choices_to_str(pinyin_choices, sep=sep)
        return pinyin_str

    def convert(self, text: str, sep: str = "") -> str:
        """alias of `text_to_pinyin_str`"""
        return self.text_to_pinyin_str(text, sep=sep)


if __name__ == "__main__":
    from tclogger import logger, logstr

    texts = [
        *["長城", "changcheng", "长亭外长"],
        *["长大", "校长", "冒顿单于"],
        *["暖和", "秋葉aaaki"],
    ]

    for text in texts:
        logger.note(f"> {logstr.mesg(text)}：", end=" ")
        pinyinizer = ChinesePinyinizer()
        pinyin = pinyinizer.convert(text, sep=" ")
        logger.success(f"{pinyin}")
        pinyin_combinations = pinyinizer.text_to_pinyin_combinations(text)
        logger.success(f"{pinyin_combinations}")

    # python -m converters.query.pinyin
