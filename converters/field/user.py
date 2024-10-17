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
        if val.strip():
            val_int = int(val.strip())
            res = {"owner.mid": val_int}
        return res

    def list_val_to_es_dict(self, vals: str) -> dict:
        """
        - {'field':'uid', 'field_type':'uid', 'op':'=',
            'vals': '642389251,946974,1780480185', 'val_type':'list'}
            -> {"terms": {"uid": [642389251,946974,1780480185]}}
        """
        res = {}
        if vals:
            vals_ints = [int(val.strip()) for val in vals.split(",") if val.strip()]
            if vals_ints:
                res = {"owner.mid": vals_ints}
        return res

    def filter_dict_to_es_dict(self, filter_dict: dict) -> dict:
        res = {}
        if filter_dict["val_type"] == "value":
            res = self.op_val_to_es_dict(filter_dict["val"])
        elif filter_dict["val_type"] == "list":
            res = self.list_val_to_es_dict(filter_dict["vals"])
        else:
            logger.warn(f"× No matching val type: {filter_dict['val_type']}")
        return res
