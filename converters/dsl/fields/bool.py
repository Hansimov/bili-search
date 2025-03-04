from collections import defaultdict

from converters.dsl.constants import ES_BOOL_OPS, ES_BOOL_OP_TYPE, MSM


class BoolElasticReducer:
    def get_bool_op(self, bool_clause: dict) -> ES_BOOL_OP_TYPE:
        """Get bool key of bool_clause, which is one of:
        - {"bool": {<bool_op>: <bool_dict>}}
            - "must", "should", "must_not"
        - {"filter": <filter_dict>}
            - "filter"
        """
        if "bool" in bool_clause:
            return next(iter(bool_clause["bool"]))
        if "filter" in bool_clause:
            return "filter"
        return None

    def get_bool_dict(
        self,
        bool_clause: dict,
        bool_op: ES_BOOL_OP_TYPE = None,
    ) -> dict:
        """Get bool dict of bool_clause,
        - {"bool": {<bool_op>: <bool_dict>}}
        """
        if not bool_op:
            bool_op = self.get_bool_op(bool_clause)
        if bool_op:
            if bool_op == "filter":
                return bool_clause.get(bool_op, {})
            else:
                return bool_clause.get("bool", {}).get(bool_op, {})

    def get_bool_op_and_dict(self, bool_clause: dict) -> tuple[ES_BOOL_OP_TYPE, dict]:
        """Get bool op and dict of bool_clause"""
        bool_op = self.get_bool_op(bool_clause)
        bool_dict = self.get_bool_dict(bool_clause, bool_op)
        return bool_op, bool_dict

    def get_bool_ops_and_dicts(
        self, bool_clause: dict
    ) -> list[tuple[ES_BOOL_OP_TYPE, dict]]:
        """Get list of bool ops and dicts of bool_clause"""
        bool_ops_and_dicts = []
        bool_items = bool_clause.get("bool", {})
        for bool_op, bool_dict in bool_items.items():
            if bool_dict:
                bool_ops_and_dicts.append((bool_op, bool_dict))
        filter_dict = bool_clause.get("filter", {})
        if filter_dict:
            bool_ops_and_dicts.append(("filter", filter_dict))
        return bool_ops_and_dicts

    def get_minimum_should_match(self, bool_clause: dict) -> int:
        if bool_clause.get("bool", {}).get("should"):
            return bool_clause["bool"].get(MSM, None)

    def sort_op_list_dict(self, op_list_dict: dict) -> dict:
        """Sort op_list_dict keys by:
        - must, must_not, filter, should
        - others keys are sorted by string comparison
        """
        sorted_keys = sorted(
            op_list_dict.keys(),
            key=lambda x: str(ES_BOOL_OPS.index(x)) if x in ES_BOOL_OPS else x,
        )
        op_list_dict = dict(
            sorted(op_list_dict.items(), key=lambda x: sorted_keys.index(x[0]))
        )
        return op_list_dict

    def reduce_bool_clauses(self, bool_clauses: list[dict], sort: bool = True) -> dict:
        """Reduce list of bool clauses to single bool clause"""
        if not bool_clauses:
            return {}
        if len(bool_clauses) == 1:
            return bool_clauses[0]
        op_list_dict = defaultdict(list)
        for bool_clause in bool_clauses:
            bool_ops_and_dict = self.get_bool_ops_and_dicts(bool_clause)
            if bool_ops_and_dict:
                for bool_op, bool_dict in bool_ops_and_dict:
                    op_list_dict[bool_op].append(bool_dict)
                    if bool_op == "should":
                        min_should = self.get_minimum_should_match(bool_clause)
                        if min_should is not None:
                            op_list_dict[MSM].append(min_should)
        for bool_op, bool_dict_list in op_list_dict.items():
            if bool_op == MSM:
                op_list_dict[MSM] = min(bool_dict_list)
            if len(bool_dict_list) == 1:
                op_list_dict[bool_op] = bool_dict_list[0]
        if sort:
            op_list_dict = self.sort_op_list_dict(op_list_dict)
        return {"bool": op_list_dict}

    def reduce(self, elastic_dict: dict) -> dict:
        """Reduce bool queries in elastic dict"""
        return elastic_dict
