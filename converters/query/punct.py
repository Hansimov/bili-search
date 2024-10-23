import string


class Puncter:
    def __init__(
        self,
        encoding: str = "gb2312",
        is_whitespace_punct: bool = False,
        non_specials: str = "",
        specials: str = "",
    ):
        self.encoding = encoding
        self.en_puncts = string.punctuation
        if is_whitespace_punct:
            self.en_puncts += string.whitespace
        self.non_specials = non_specials
        self.specials = specials

    def is_special_char(self, ch: str) -> bool:
        """GB2312 编码表：https://www.toolhelper.cn/Encoding/GB2312"""
        if not ch or len(ch) > 1:
            raise ValueError("Invalid character")
        if self.specials and ch in self.specials:
            return True
        if self.non_specials and ch in self.non_specials:
            return False
        bytes = ch.encode(self.encoding, errors="ignore")
        if len(bytes) == 1:
            return ch in self.en_puncts
        elif len(bytes) == 2:
            byte1, byte2 = bytes
            # A1A0~A3FE, A6A0~A9FE (treat japanese (A4A0~A5FE) as non-special)
            if (byte1 >= 0xA1 and byte1 <= 0xA3) or (byte1 >= 0xA6 and byte1 <= 0xA9):
                return True
        return False

    def is_contain_special(self, text: str) -> bool:
        for ch in text:
            if self.is_special_char(ch):
                return True
        return False

    def remove(self, text: str) -> str:
        text = "".join(ch for ch in text if not self.is_special_char(ch))
        return text


class HansChecker:
    def __init__(self):
        self.puncter = Puncter(specials=string.printable)

    def no_hans(self, text: str) -> bool:
        for ch in text:
            if not self.puncter.is_special_char(ch):
                return False
        return True


if __name__ == "__main__":
    from tclogger import logger

    text = "Hello, world! 你好，世界！ @-我_のなまえは!"
    puncter = Puncter(non_specials="-_")
    logger.note(f"> {text}:")
    logger.success(f"{puncter.remove(text)}")

    texts = ["のなまえは", "hello World!", "你好 world", "are you ok 123？"]
    checker = HansChecker()
    logger.note(f"Checking has no hans:")
    for text in texts:
        logger.note(f"> {text}:", end=" ")
        logger.success(f"{checker.no_hans(text)}")

    # python -m converters.query.punct
