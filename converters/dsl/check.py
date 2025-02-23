from lark import Lark
from pathlib import Path
from tclogger import logger


class DslSyntaxChecker:
    def __init__(self, verbose: bool = False):
        self.dsl_lark = Path(__file__).parent / "syntax.lark"
        self.verbose = verbose
        self.init_parser()

    def init_parser(self):
        with open(self.dsl_lark, "r") as rf:
            syntax = rf.read()
        self.parser = Lark(syntax, parser="earley")

    def check(self, expr: str) -> tuple[bool, any]:
        try:
            res = self.parser.parse(expr)
            if self.verbose:
                logger.success(f"✓ {expr}")
            return res
        except Exception as e:
            if self.verbose:
                logger.warn(f"× {expr}")
                raise e
            return None


def test_syntax_checker():
    from converters.dsl.test import queries

    checker = DslSyntaxChecker(verbose=True)
    for query in queries:
        res = checker.check(query)
        logger.mesg(res.pretty(), verbose=bool(res))


if __name__ == "__main__":
    test_syntax_checker()

    # python -m converters.dsl.check
