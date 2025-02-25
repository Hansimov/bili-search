from tclogger import logger, dict_to_str

from converters.dsl.parse import DslLarkParser
from converters.dsl.node import DslExprNode, DslTreeBuilder, DslTreeExprGrouper
from converters.dsl.fields.date import DateExprElasticConverter
from converters.dsl.node import ATOMS


class DslExprToElasticConverter(DslTreeExprGrouper):
    def convert(self, expr: str) -> dict:
        parser = DslLarkParser()
        lark_tree = parser.parse(expr)
        # logger.mesg(parser.str(lark_tree), verbose=lark_tree)

        builder = DslTreeBuilder()
        dsl_tree = builder.build_dsl_tree_from_lark_tree(lark_tree)
        # logger.mesg(str(dsl_tree), verbose=dsl_tree)

        grouper = DslTreeExprGrouper()
        expr_tree = grouper.group_dsl_tree_to_expr_tree(dsl_tree)
        logger.mesg(str(expr_tree), verbose=expr_tree)

        elastic_dict = self.expr_node_to_elastic_dict(expr_tree)
        logger.mesg(dict_to_str(elastic_dict), indent=2)
        if not elastic_dict:
            raise NotImplementedError(f"× Not implemented: {expr}")

        return elastic_dict

    def expr_node_to_elastic_dict(self, node: DslExprNode) -> dict:
        expr_node = node.find_child_with_key(ATOMS)
        if expr_node.key == "date_expr":
            return DateExprElasticConverter().convert(node)


def test_date_field():
    from converters.field.test import test_date_strs
    from converters.field.date import DateFieldConverter

    field_converter = DateFieldConverter()
    elastic_converter = DslExprToElasticConverter()
    for date_str in test_date_strs[:]:
        logger.note(f"{date_str}")
        date_expr = f"date={date_str}"
        elastic_dict = elastic_converter.convert(date_expr)


if __name__ == "__main__":
    test_date_field()

    # python -m converters.dsl.elastic
