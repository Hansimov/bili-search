from collections import defaultdict

from converters.dsl.constants import ES_BOOL_OPS, ES_BOOL_OP_TYPE, MSM


class BoolElasticReducer:
    def get_bool_op(self, bool_clause: dict) -> ES_BOOL_OP_TYPE:
        """Get bool key of bool_clause, which is one of:
        - {"bool": {<bool_op>: <bool_dict>}}
            - "must", "should", "must_not", "filter"
        """
        if "bool" in bool_clause:
            return next(iter(bool_clause["bool"]))
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
            return bool_clause.get("bool", {}).get(bool_op, {})

    def get_bool_op_and_dict(self, bool_clause: dict) -> tuple[ES_BOOL_OP_TYPE, dict]:
        """Get bool op and dict of bool_clause"""
        bool_op = self.get_bool_op(bool_clause)
        bool_dict = self.get_bool_dict(bool_clause, bool_op)
        return bool_op, bool_dict

    def get_bool_ops_and_dicts(
        self, bool_clause: dict
    ) -> list[tuple[ES_BOOL_OP_TYPE, dict]]:
        """Return list of tuples: [(bool_op, bool_dict), ...]"""
        bool_ops_and_dicts = []
        bool_items = bool_clause.get("bool", {})
        for bool_op, bool_dict in bool_items.items():
            if bool_dict is not None:
                bool_ops_and_dicts.append((bool_op, bool_dict))
        return bool_ops_and_dicts

    def get_minimum_should_match(self, bool_clause: dict) -> int:
        if bool_clause.get("bool", {}).get("should"):
            return bool_clause["bool"].get(MSM, None)

    def combine_zero_msm_clauses(self, should_clauses: list[dict]) -> list[dict]:
        """Input: list of: {"should": ..., "minimum_should_match": 0}
        Output: {"should": [...]}"""
        return [should_clause["should"] for should_clause in should_clauses]

    def combine_non_zero_msm_clauses(self, should_clauses: list[dict]) -> list[dict]:
        """Input: list of: {"should": ..., "minimum_should_match": X}
        Output: {"should": [...]}"""
        return [
            {"bool": {"should": should_clause["should"], MSM: should_clause[MSM]}}
            for should_clause in should_clauses
        ]

    def add_shoulds_to_op_list_dict(
        self, op_list_dict: dict, should_clauses: list[dict]
    ) -> dict:
        if len(should_clauses) == 0:
            return op_list_dict
        if len(should_clauses) == 1:
            op_list_dict.update(should_clauses[0])
            return op_list_dict
        zero_msm_clauses = []
        non_zero_msm_clauses = []
        for should_clause in should_clauses:
            if should_clause.get(MSM, None) == 0:
                zero_msm_clauses.append(should_clause)
            else:
                non_zero_msm_clauses.append(should_clause)
        if zero_msm_clauses:
            op_list_dict["should"] = self.combine_zero_msm_clauses(zero_msm_clauses)
            op_list_dict[MSM] = 0
        op_list_dict["must"].extend(
            self.combine_non_zero_msm_clauses(non_zero_msm_clauses)
        )
        return op_list_dict

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

    def reduce_co_bool_clauses(
        self, bool_clauses: list[dict], sort: bool = True
    ) -> dict:
        """Return a single bool clause dict, which is reduced from list of bool clauses (under `co` or `and`)"""
        if not bool_clauses:
            return {}
        if len(bool_clauses) == 1:
            return bool_clauses[0]
        op_list_dict = defaultdict(list)
        should_clauses = []
        for bool_clause in bool_clauses:
            bool_ops_and_dict = self.get_bool_ops_and_dicts(bool_clause)
            if not bool_ops_and_dict:
                continue
            for bool_op, bool_dict in bool_ops_and_dict:
                if bool_op == "should":
                    should_clause = {
                        "should": bool_dict,
                        MSM: self.get_minimum_should_match(bool_clause),
                    }
                    should_clauses.append(should_clause)
                elif bool_op == MSM:
                    continue
                else:
                    op_list_dict[bool_op].append(bool_dict)
        op_list_dict = self.add_shoulds_to_op_list_dict(op_list_dict, should_clauses)
        for bool_op, bool_dict_list in op_list_dict.items():
            if isinstance(bool_dict_list, list) and len(bool_dict_list) == 1:
                op_list_dict[bool_op] = bool_dict_list[0]
        if sort:
            op_list_dict = self.sort_op_list_dict(op_list_dict)
        return {"bool": op_list_dict}

    def reduce(self, elastic_dict: dict) -> dict:
        """Reduce bool queries in elastic dict"""
        return elastic_dict
