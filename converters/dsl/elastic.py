from tclogger import logger, dict_to_str

from converters.dsl.parse import DslLarkParser
from converters.dsl.node import DslExprNode, DslTreeBuilder, DslTreeExprGrouper
from converters.dsl.fields.date import DateExprElasticConverter


class DslExprToElasticConverter(DslTreeExprGrouper):
    def convert(self, expr: str) -> dict:
        self.tree_dict = {}

        parser = DslLarkParser()
        lark_tree = parser.parse(expr)
        # logger.mesg(parser.str(lark_tree), verbose=lark_tree)

        builder = DslTreeBuilder()
        dsl_tree = builder.build_dsl_tree_from_lark_tree(lark_tree)
        # logger.mesg(str(dsl_tree), verbose=dsl_tree)

        grouper = DslTreeExprGrouper()
        expr_tree = grouper.group_dsl_tree_to_expr_tree(dsl_tree)
        logger.mesg(str(expr_tree), verbose=expr_tree)
        return self.tree_dict

    def expr_node_to_elastic_dict(self, node: DslExprNode) -> dict:
        children = node.children
        if self.is_start(node):
            return self.expr_node_to_elastic_dict(children[0])
        elif self.is_atom(node):
            return self.expr_node_to_elastic_dict(children[0])
        elif node.key == "date_expr":
            return DateExprElasticConverter().convert(node)
        else:
            return None


def test_dsl_to_elastic_converter():
    from converters.dsl.test import queries

    converter = DslExprToElasticConverter(verbose=True)

    for query in queries[-1:]:
        tree_dict = converter.convert(query)
        # logger.mesg(dict_to_str(tree_dict))


def test_date_field():
    from converters.field.test import test_date_strs
    from converters.field.date import DateFieldConverter

    field_converter = DateFieldConverter()
    elastic_converter = DslExprToElasticConverter()
    for date_str in test_date_strs[:3]:
        logger.note(f"{date_str}")
        date_expr = f"date={date_str}"
        elastic_dict = elastic_converter.convert(date_expr)
        logger.mesg(dict_to_str(elastic_dict))


if __name__ == "__main__":
    # test_dsl_to_elastic_converter()
    test_date_field()

    # python -m converters.dsl.elastic
