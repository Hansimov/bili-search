import re

from tclogger import logger
from converters.field.operators import RE_COMMA
from converters.query.punct import Puncter


class UserFieldConverter:
    RE_USER_FIELD = r"(name|author|uploader|up|user)"
    REP_USER_FIELD = rf"(?P<user_field>{RE_USER_FIELD})"
    RE_USER_VAL = r"[^:：,，\[\]\(\)\n\s]+"
    # indeed, bilibili user names only support "-" and "_"

    def __init__(self):
        self.puncter = Puncter()
        self.field = "owner.name.keyword"

    def validate_val(self, val: str) -> str:
        if val.strip():
            val_str = self.puncter.remove(val.strip())
            if val_str:
                return val_str
        return None

    def op_val_to_es_dict(self, val: str) -> dict:
        """
        - {'field':'user', 'field_type':'user', 'op':'=', 'val':'影视飓风', 'val_type':'value'}
            -> {"term": {"uid": "影视飓风"}}
        """
        res = {}
        val_str = self.validate_val(val)
        if val_str:
            res = {self.field: val_str}
        return res

    def list_val_to_es_dict(self, vals: str) -> dict:
        """
        - {'field':'user', 'field_type':'user', 'op':'=',
            'vals': '影视飓风,飓多多StormCrew,亿点点不一样', 'val_type':'list'}
            -> {"terms": {"uid": ["影视飓风","飓多多StormCrew","亿点点不一样"]}}
        """
        res = {}
        if vals:
            vals_strs = []
            for val in re.split(RE_COMMA, vals):
                val_str = self.validate_val(val)
                if val_str:
                    vals_strs.append(val_str)
            if vals_strs:
                res = {self.field: vals_strs}
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


class UidFieldConverter:
    RE_UID_FIELD = r"(mid|uid)"
    REP_UID_FIELD = rf"(?P<uid_field>{RE_UID_FIELD})"
    RE_UID_VAL = rf"\d+"

    def __init__(self):
        self.field = "owner.mid"

    def op_val_to_es_dict(self, val: str) -> dict:
        """
        - {'field':'uid', 'field_type':'uid', 'op':'=', 'val':'642389251', 'val_type':'value'}
            -> {"term": {"uid": 642389251}}
        """
        res = {}
        if val.strip():
            val_int = int(val.strip())
            res = {self.field: val_int}
        return res

    def list_val_to_es_dict(self, vals: str) -> dict:
        """
        - {'field':'uid', 'field_type':'uid', 'op':'=',
            'vals': '642389251,946974,1780480185', 'val_type':'list'}
            -> {"terms": {"uid": [642389251,946974,1780480185]}}
        """
        res = {}
        if vals:
            vals_ints = [
                int(val.strip()) for val in re.split(RE_COMMA, vals) if val.strip()
            ]
            if vals_ints:
                res = {self.field: vals_ints}
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
