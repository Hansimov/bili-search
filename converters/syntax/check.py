from lark import Lark
from pathlib import Path
from tclogger import logger


class DslSyntaxChecker:
    def __init__(self, verbose: bool = False):
        self.dsl_lark = Path(__file__).parent / "dsl.lark"
        self.verbose = verbose
        self.init_parser()

    def init_parser(self):
        with open(self.dsl_lark, "r") as rf:
            syntax = rf.read()
        self.parser = Lark(syntax, start="bles", parser="earley")

    def check(self, expr: str) -> tuple[bool, any]:
        try:
            res = self.parser.parse(expr)
            if self.verbose:
                logger.success(f"✓ {expr}")
            return res
        except Exception as e:
            if self.verbose:
                print(e)
                logger.warn(f"× {expr}")
            return None


def test_syntax_checker():
    from converters.syntax.test import queries

    checker = DslSyntaxChecker(verbose=True)
    for query in queries:
        res = checker.check(query)
        if res:
            logger.mesg(res)


if __name__ == "__main__":
    test_syntax_checker()

    # python -m converters.syntax.check
