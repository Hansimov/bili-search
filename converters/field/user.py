import re

from tclogger import logger
from converters.field.operators import OP_MAP, BRACKET_MAP


class UserFieldConverter:
    RE_USER_FIELD = r"(name|author|uploader|up|user)"
    REP_USER_FIELD = rf"(?P<user_field>{RE_USER_FIELD})"
    RE_USER_VAL = r"[^:\n\s\.]+"  # indeed, bilibili user names only support "-" and "_"


class UidFieldConverter:
    RE_UID_FIELD = r"(mid|uid)"
    REP_UID_FIELD = rf"(?P<uid_field>{RE_UID_FIELD})"
    RE_UID_VAL = rf"\d+"

    def op_val_to_es_dict(self, val: str) -> dict:
        """
        - {'field':'uid', 'field_type':'uid', 'op':'=', 'val':'642389251', 'val_type':'value'}
            -> {"term": {"uid": 642389251}}
        """
        res = {}
        if val:
            val_int = int(val)
            res = {"owner.mid": val_int}
        return res

    def list_val_to_es_dict(self, lval: str, rvals: str) -> dict:
        """
        - {'field':'uid', 'field_type':'uid', 'op':'=',
            'lval':'642389251', 'rvals': ',946974,1780480185', 'val_type':'list'}
            -> {"terms": {"uid": [642389251,946974,1780480185]}}
        """
        res = {}
        if lval:
            lval_ints = [int(lval)]
        else:
            lval_ints = []
        if rvals:
            rvals_ints = [int(val.strip()) for val in rvals.split(",") if val.strip()]
        else:
            rvals_ints = []

        val_ints = [*lval_ints, *rvals_ints]
        if val_ints:
            res = {"owner.mid": val_ints}
        return res

    def filter_dict_to_es_dict(self, filter_dict: dict) -> dict:
        res = {}
        if filter_dict["val_type"] == "value":
            res = self.op_val_to_es_dict(filter_dict["val"])
        elif filter_dict["val_type"] == "list":
            res = self.list_val_to_es_dict(filter_dict["lval"], filter_dict["rvals"])
        else:
            logger.warn(f"× No matching val type: {filter_dict['val_type']}")
        return res
