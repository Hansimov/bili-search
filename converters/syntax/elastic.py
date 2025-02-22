from tclogger import logger

from converters.syntax.dsl import DslLarkParser
from converters.syntax.node import DslTreeBuilder, DslTreeExprGrouper


class DslToElasticConverter:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def construct(self, expr: str) -> dict:
        self.tree_dict = {}

        parser = DslLarkParser()
        lark_tree = parser.parse(expr)
        logger.mesg(parser.str(lark_tree), verbose=lark_tree)

        builder = DslTreeBuilder()
        dsl_tree = builder.build_dsl_tree_from_lark_tree(lark_tree)
        logger.mesg(str(dsl_tree), verbose=dsl_tree)

        grouper = DslTreeExprGrouper()
        expr_tree = grouper.group_dsl_tree_to_expr_tree(dsl_tree)
        logger.mesg(str(expr_tree), verbose=expr_tree)
        return self.tree_dict


def test_dsl_to_elastic_converter():
    from converters.syntax.test import queries

    converter = DslToElasticConverter(verbose=True)

    for query in queries[-1:]:
        tree_dict = converter.construct(query)
        # logger.mesg(dict_to_str(tree_dict))


if __name__ == "__main__":
    test_dsl_to_elastic_converter()

    # python -m converters.syntax.elastic
