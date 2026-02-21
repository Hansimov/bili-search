from dsl.elastic import DslExprToElasticConverter
from dsl.filter import QueryDslDictFilterMerger
from dsl.rewrite import DslExprRewriter
from tclogger import dict_to_str

rewriter = DslExprRewriter()
converter = DslExprToElasticConverter()
merger = QueryDslDictFilterMerger()

query = "搜索词 u!=[影视飓风,何同学]"
query_info = rewriter.get_query_info(query)
expr_tree = query_info["query_expr_tree"]

filter_expr_tree = expr_tree.filter_atoms_by_keys(exclude_keys=["word_expr"])
filter_dict = converter.expr_tree_to_dict(filter_expr_tree)

# Old method (bug): only extracts bool.filter, ignoring must_not
old_filter_clauses = merger.get_query_filters_from_query_dsl_dict(filter_dict)
print("=== Old method (only filter) ===")
print(f"filter_clauses: {old_filter_clauses}")

# New method (fix): extracts filter + must_not
new_filter_clauses = merger.get_all_bool_clauses_from_query_dsl_dict(filter_dict)
print()
print("=== New method (filter + must_not) ===")
for clause in new_filter_clauses:
    print(dict_to_str(clause, add_quotes=True))

# Test with combined filters
query2 = "搜索词 u!=[影视飓风,何同学] v>10k"
query_info2 = rewriter.get_query_info(query2)
expr_tree2 = query_info2["query_expr_tree"]
filter_expr_tree2 = expr_tree2.filter_atoms_by_keys(exclude_keys=["word_expr"])
filter_dict2 = converter.expr_tree_to_dict(filter_expr_tree2)
new_filter_clauses2 = merger.get_all_bool_clauses_from_query_dsl_dict(filter_dict2)
print()
print("=== Combined: user exclusion + stat filter ===")
for clause in new_filter_clauses2:
    print(dict_to_str(clause, add_quotes=True))
