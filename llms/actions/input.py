class InputCleaner:
    def __init__(self):
        pass

    def de_backtick(self, text: str):
        return text.strip().strip("`").strip()

    def clean(self, text: str) -> str:
        text = self.de_backtick(text)
        return text
