import string


class Puncter:
    def __init__(self, encoding: str = "gb2312", is_whitespace_punct: bool = False):
        self.encoding = encoding
        self.en_puncts = string.punctuation
        if is_whitespace_punct:
            self.en_puncts += string.whitespace

    def is_special_char(self, ch: str, non_specials: str = "") -> bool:
        """GB2312 编码表：https://www.toolhelper.cn/Encoding/GB2312"""
        if not ch or len(ch) > 1:
            raise ValueError("Invalid character")
        if non_specials and ch in non_specials:
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

    def is_contain_special(self, text: str, non_specials: str = ":：") -> bool:
        for ch in text:
            if self.is_special_char(ch, non_specials=non_specials):
                return True

        return False

    def remove(self, text: str) -> str:
        text = "".join([ch for ch in text if not self.is_special_char(ch)])
        return text


if __name__ == "__main__":
    from tclogger import logger

    text = "Hello, world!  你好，世界！ @我のなまえは!"
    puncter = Puncter()
    logger.note(f"> {text}：")
    logger.success(f"{puncter.remove(text)}")

    # python -m converters.query.punct
