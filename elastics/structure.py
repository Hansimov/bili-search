from copy import deepcopy
from tclogger import logger
from typing import Union

from converters.query.field import is_pinyin_field, deboost_field, boost_fields
from converters.query.field import remove_suffixes_from_fields
from elastics.videos.constants import SEARCH_MATCH_FIELDS, SEARCH_BOOSTED_FIELDS
from elastics.videos.constants import DATE_MATCH_FIELDS, DATE_BOOSTED_FIELDS


def get_es_source_val(d: dict, key: str):
    keys = key.split(".")
    dd = deepcopy(d)
    for key in keys:
        if isinstance(dd, dict) and key in dd:
            dd = dd[key]
        else:
            joined_key = ".".join(keys)
            if joined_key in dd:
                return dd[joined_key]
            else:
                return None

    return dd


def get_highlight_settings(
    match_fields: list[str],
    removable_suffixes: list[str] = [".words"],
    tag: str = "hit",
):
    highlight_fields = [
        deboost_field(field) for field in match_fields if not is_pinyin_field(field)
    ]
    if removable_suffixes:
        highlight_fields.extend(
            remove_suffixes_from_fields(highlight_fields, suffixes=removable_suffixes)
        )

    highlight_fields = sorted(list(set(highlight_fields)))
    highlight_fields_dict = {field: {} for field in highlight_fields}

    highlight_settings = {
        "pre_tags": [f"<{tag}>"],
        "post_tags": [f"</{tag}>"],
        "fields": highlight_fields_dict,
    }
    return highlight_settings


def construct_boosted_fields(
    match_fields: list[str] = SEARCH_MATCH_FIELDS,
    boost: bool = True,
    boosted_fields: dict = SEARCH_BOOSTED_FIELDS,
    use_pinyin: bool = False,
) -> tuple[list[str], list[str]]:
    if not use_pinyin:
        match_fields = [
            field for field in match_fields if not field.endswith(".pinyin")
        ]
    date_fields = [
        field
        for field in match_fields
        if not field.endswith(".pinyin")
        and any(field.startswith(date_field) for date_field in DATE_MATCH_FIELDS)
    ]
    if boost:
        boosted_match_fields = boost_fields(match_fields, boosted_fields)
        boosted_date_fields = boost_fields(date_fields, DATE_BOOSTED_FIELDS)
    else:
        boosted_match_fields = match_fields
        boosted_date_fields = date_fields
    return boosted_match_fields, boosted_date_fields


def set_timeout(body: dict, timeout: Union[int, float, str] = None):
    if timeout is not None:
        if isinstance(timeout, str):
            body["timeout"] = timeout
        elif isinstance(timeout, (int, float)):
            timeout_str = round(timeout * 1000)
            body["timeout"] = f"{timeout_str}ms"
        else:
            logger.warn(f"× Invalid type of `timeout`: {type(timeout)}")
    return body


def set_min_score(body: dict, min_score: float = None):
    if min_score is not None:
        body["min_score"] = min_score
    return body


def set_terminate_after(body: dict, terminate_after: int = None):
    if terminate_after is not None:
        body["terminate_after"] = terminate_after
    return body


if __name__ == "__main__":
    d = {
        "owner": {"name": {"space": "value1"}},
        "stat": {"rights": {"view": "value2"}},
        "pubdate.time": "value3",
    }

    k1 = "owner.name.space"
    k2 = "stat.rights.view"
    k3 = "pubdate.time"
    for k in [k1, k2, k3]:
        print(f"{k}: {get_es_source_val(d,k)}")

    # python -m elastics.structure
