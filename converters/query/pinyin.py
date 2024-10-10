import pypinyin
import zhconv


class ChinesePinyinizer:
    def text_to_simple(self, text: str) -> str:
        return zhconv.convert(text, "zh-cn")

    def text_to_pinyin_segs(self, text: str) -> list[list[str]]:
        return pypinyin.pinyin(text, style=pypinyin.STYLE_NORMAL, heteronym=True)

    def pinyin_segs_to_str(self, tlist: list[list[str]], sep: str = "") -> str:
        return sep.join([t[0] for t in tlist])

    def convert(self, text: str, sep: str = "") -> str:
        text = self.text_to_simple(text)
        pinyin_segs = self.text_to_pinyin_segs(text)
        pinyin_str = self.pinyin_segs_to_str(pinyin_segs, sep=sep)
        return pinyin_str


if __name__ == "__main__":
    from tclogger import logger, logstr

    texts = ["秋葉aaaki", "長城", "长亭外长", "长大", "校长", "冒顿单于", "暖和"]

    for text in texts:
        logger.note(f"> {logstr.mesg(text)}：", end=" ")
        pinyinizer = ChinesePinyinizer()
        pinyin = pinyinizer.convert(text)
        logger.success(f"{pinyin}")

    # python -m converters.query.pinyin
