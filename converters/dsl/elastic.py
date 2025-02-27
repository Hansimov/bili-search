from tclogger import logger, dict_to_str

from converters.dsl.parse import DslLarkParser
from converters.dsl.node import DslExprNode, DslTreeBuilder, DslTreeExprGrouper
from converters.dsl.fields.date import DateExprElasticConverter
from converters.dsl.fields.user import UserExprElasticConverter
from converters.dsl.node import ATOMS


class DslExprToElasticConverter(DslTreeExprGrouper):
    def __init__(self):
        self.parser = DslLarkParser()
        self.builder = DslTreeBuilder()
        self.grouper = DslTreeExprGrouper()

    def expr_node_to_elastic_dict(self, node: DslExprNode) -> dict:
        expr_node = node.find_child_with_key(ATOMS)
        if expr_node.is_key("date_expr"):
            return DateExprElasticConverter().convert(node)
        elif expr_node.is_key("user_expr"):
            return UserExprElasticConverter().convert(node)

    def convert(self, expr: str) -> dict:
        lark_tree = self.parser.parse(expr)
        # logger.mesg(parser.str(lark_tree), verbose=lark_tree)
        dsl_tree = self.builder.build_dsl_tree_from_lark_tree(lark_tree)
        # logger.mesg(str(dsl_tree), verbose=dsl_tree)
        expr_tree = self.grouper.group_dsl_tree_to_expr_tree(dsl_tree)
        logger.mesg(str(expr_tree), verbose=expr_tree)
        elastic_dict = self.expr_node_to_elastic_dict(expr_tree)
        return elastic_dict


def test_date_field():
    from converters.field.test import test_date_strs, test_date_list_strs
    from converters.field.date import DateFieldConverter

    field_converter = DateFieldConverter()
    elastic_converter = DslExprToElasticConverter()
    for date_str in test_date_list_strs[:]:
        logger.note(f"{date_str}")
        date_expr = f"date={date_str}"
        elastic_dict = elastic_converter.convert(date_expr)
        logger.mesg(dict_to_str(elastic_dict), indent=2)


def test_user_field():
    from converters.field.test import test_user_strs

    elastic_converter = DslExprToElasticConverter()
    for user_str in test_user_strs:
        logger.note(f"{user_str}")
        elastic_dict = elastic_converter.convert(user_str)
        logger.mesg(dict_to_str(elastic_dict), indent=2)


if __name__ == "__main__":
    # test_date_field()
    test_user_field()

    # python -m converters.dsl.elastic
