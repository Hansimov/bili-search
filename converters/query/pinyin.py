import pypinyin
import zhconv

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
        text = self.text_to_simple(text)
        return pypinyin.pinyin(text, style=pypinyin.STYLE_NORMAL, heteronym=True)

    def text_to_pinyin_segs(self, text: str) -> list[str]:
        pinyin_choices = self.text_to_pinyin_choices(text)
        pinyin_segs = [choice[0] for choice in pinyin_choices]
        return pinyin_segs

    def text_to_pinyin_str(self, text: str, sep: str = "") -> str:
        pinyin_segs = self.text_to_pinyin_segs(text)
        pinyin_str = sep.join(pinyin_segs)
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
        pinyin_str = pinyinizer.convert(text, sep=" ")
        logger.success(f"{pinyin_str}")
        pinyin_segs = pinyinizer.text_to_pinyin_segs(text)
        logger.success(f"{pinyin_segs}")

    # python -m converters.query.pinyin
