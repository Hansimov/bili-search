import re

from copy import deepcopy
from typing import Union, Literal


def deboost_field(field: str):
    return field.split("^", 1)[0]


def boost_fields(match_fields: list, boosted_fields: dict):
    boosted_match_fields = deepcopy(match_fields)
    for key in boosted_fields:
        if key in boosted_match_fields:
            key_index = boosted_match_fields.index(key)
            boosted_match_fields[key_index] += f"^{boosted_fields[key]}"
    return boosted_match_fields


def is_pinyin_field(field: str):
    return deboost_field(field).endswith(".pinyin")


def is_field_in_fields(field_to_check: str, fields: list[str]) -> bool:
    for field in fields:
        if field.startswith(field_to_check):
            return True
    return False


def remove_boost_from_fields(fields: list[str]) -> list[str]:
    return [field.split("^", 1)[0] for field in fields]


def remove_fields_from_fields(
    fields_to_remove: Union[str, list], fields: list[str]
) -> list[str]:
    if isinstance(fields_to_remove, str):
        fields_to_remove = [fields_to_remove]
    clean_fields = []
    for field in fields:
        for field_to_remove in fields_to_remove:
            if not field.startswith(field_to_remove):
                clean_fields.append(field)
    return clean_fields


RE_FIELD_SUFFIX = (
    r"^(?P<field>.+?)(?P<suffix>\.({}))?(?P<boost_str>\^(?P<boost>\d+(\.\d+)?))?$"
)
RE_FIELD_SUFFIX_WORDS = RE_FIELD_SUFFIX.format("words")
RE_COMPILE_SUFFIX_WORDS = re.compile(RE_FIELD_SUFFIX_WORDS)


def is_field_ends_with(
    field: str,
    suffixes: list[Literal[".words", ".pinyin"]] = [],
    re_compile: re.Pattern = None,
):
    if re_compile:
        match = re_compile.match(field)
    else:
        re_suffixes = "|".join(suffix.lstrip(".") for suffix in suffixes)
        pattern = RE_FIELD_SUFFIX.format(re_suffixes)
        match = re.match(pattern, field)
    if match:
        field_name = match.group("field")
        field_boost = match.group("boost") or ""
        field_suffix = match.group("suffix") or ""
        bool_flag = not (field_suffix == "")
        return bool_flag, field_name, field_suffix, field_boost
    else:
        return False, field, "", ""


def build_boosts_from_fields(fields: list[str]):
    boosts = {}
    for field in fields:
        bool_flag, field_name, field_suffix, field_boost = is_field_ends_with(field, [])
        boosts[field_name] = field_boost
    return boosts


def remove_suffixes_from_fields(
    fields: list[str],
    suffixes: list[Literal[".words", ".pinyin"]] = [],
    re_compile: re.Pattern = RE_COMPILE_SUFFIX_WORDS,
    boosts: dict = {},
):
    """Example: title.words^4 -> title^4"""
    res = []
    field_boosts = {}
    if not boosts:
        boosts = build_boosts_from_fields(fields)

    for field in fields:
        bool_flag, field_name, field_suffix, field_boost = is_field_ends_with(
            field, suffixes, re_compile=re_compile
        )
        if bool_flag:
            if field_name in boosts:
                field_boost = boosts[field_name]
            if field_name in field_boosts:
                if float(field_boost) > float(field_boosts[field_name]):
                    field_boosts[field_name] = field_boost
            else:
                field_boosts[field_name] = field_boost
        else:
            field_boosts[field_name] = field_boost
    for field_name, field_boost in field_boosts.items():
        if field_boost:
            field_boost_str = f"^{field_boost}"
        else:
            field_boost_str = ""
        res.append(f"{field_name}{field_boost_str}")
    return res


if __name__ == "__main__":
    from tclogger import logger

    fields = [
        "title.words^1.5",
        "title^1",
        "title.pinyin^2",
        "title.pinyin^3.0",
        "owner.name",
    ]
    # boosts = {"title": 2, "title.words": 1.0, "title.pinyin": 1.5}
    boosts = build_boosts_from_fields(fields)
    logger.file(boosts)

    for field in fields:
        logger.note(f"{field}:", end=" ")
        logger.success(is_field_ends_with(field, suffixes=[".words"]))
    logger.file(remove_suffixes_from_fields(fields, suffixes=[".words"]))

    # python -m converters.query.field
