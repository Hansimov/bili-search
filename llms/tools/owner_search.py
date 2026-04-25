from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import re

from llms.tools.utils import format_google_results, format_related_owners


def _normalize_query_spaces(query: str) -> str:
    return re.sub(r"\s+", " ", str(query or "")).strip()


def _normalize_seed_values(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, int)):
        values = [values]
    return [str(value).strip() for value in values or [] if str(value).strip()]


_OWNER_SOURCE_LABELS = {
    "name": "名字匹配",
    "topic": "主题发现",
    "relation": "关系发现",
    "related_tokens": "相关作者",
    "google_space": "Google 空间页",
}
_OWNER_SOURCE_WEIGHTS = {
    "name": 1.2,
    "topic": 1.0,
    "relation": 1.05,
    "related_tokens": 1.1,
    "google_space": 0.85,
}


def _strip_owner_search_filters(text: str) -> str:
    stripped = re.sub(r"q=[^\s]+", " ", str(text or ""), flags=re.IGNORECASE)
    stripped = re.sub(r":[^\s]+", " ", stripped)
    return _normalize_query_spaces(stripped)


def _coerce_owner_search_args(args: dict) -> dict:
    normalized = dict(args or {})
    text = str(normalized.get("text", "") or "").strip()

    if normalized.get("size") is None:
        for key in ("num", "limit"):
            if normalized.get(key) is not None:
                normalized["size"] = normalized.get(key)
                break

    if not text:
        alias_candidates = [
            ("topic", "topic"),
            ("name", "name"),
            ("relation", "relation"),
        ]
        for key, inferred_mode in alias_candidates:
            candidate = str(normalized.get(key, "") or "").strip()
            if not candidate:
                continue
            text = _strip_owner_search_filters(candidate)
            normalized["text"] = text
            normalized.setdefault("mode", inferred_mode)
            break

    if not text:
        query = normalized.get("query")
        queries = normalized.get("queries")
        candidates: list[str] = []
        if isinstance(query, str):
            candidates.append(query)
        if isinstance(queries, str):
            candidates.append(queries)
        elif isinstance(queries, list):
            candidates.extend(
                str(item).strip() for item in queries if str(item or "").strip()
            )
        for candidate in candidates:
            text = _strip_owner_search_filters(candidate)
            if text:
                normalized["text"] = text
                break

    mode = str(normalized.get("mode", "auto") or "auto")
    if (
        mode == "auto"
        and text
        and any(key in normalized for key in ("query", "queries"))
    ):
        normalized["mode"] = "topic"

    return normalized


def _normalize_owner_identity_key(owner: dict) -> str:
    mid = owner.get("mid")
    if mid not in (None, ""):
        return f"mid:{mid}"
    name = re.sub(r"\s+", "", str(owner.get("name", "") or "").lower())
    return f"name:{name}" if name else ""


def _extract_google_space_candidate_name(title: str) -> str:
    name = str(title or "").strip()
    if not name:
        return ""
    for marker in ("的个人空间", "个人空间", "主页"):
        if marker in name:
            name = name.split(marker, 1)[0]
    for separator in (" - ", " -", "-", "|", "｜"):
        if separator in name:
            name = name.split(separator, 1)[0]
    return name.strip(" -_|｜·，。！？?：:[]()（）")


class OwnerSearchToolMixin:
    @staticmethod
    def _owner_source_group(
        *,
        source: str,
        owners: list[dict],
        max_hits: int,
        text: str = "",
        error: str = "",
        query: str = "",
        google_results: list[dict] | None = None,
    ) -> dict:
        payload = {
            "source": source,
            "label": _OWNER_SOURCE_LABELS.get(source, source),
            "total_owners": len(owners),
            "owners": format_related_owners(owners, max_hits=max_hits),
        }
        if text:
            payload["text"] = text
        if error:
            payload["error"] = error
        if query:
            payload["query"] = query
        if google_results is not None:
            payload["google_results"] = google_results[:max_hits]
        return payload

    @staticmethod
    def _merge_owner_source_groups(
        source_owners: list[tuple[str, list[dict]]],
    ) -> list[dict]:
        merged: dict[str, dict] = {}
        for source, owners in source_owners:
            weight = _OWNER_SOURCE_WEIGHTS.get(source, 0.8)
            for rank, owner in enumerate(owners):
                key = _normalize_owner_identity_key(owner)
                if not key:
                    continue
                item = merged.setdefault(
                    key,
                    {
                        "mid": owner.get("mid"),
                        "name": owner.get("name", ""),
                        "score": owner.get("score", 0),
                        "sources": [],
                        "face": owner.get("face", ""),
                        "sample_title": owner.get("sample_title", ""),
                        "sample_bvid": owner.get("sample_bvid", ""),
                        "sample_pic": owner.get("sample_pic", ""),
                        "sample_view": owner.get("sample_view"),
                        "_fusion_score": 0.0,
                        "_best_rank": rank,
                        "_best_owner_score": float(owner.get("score", 0) or 0),
                    },
                )
                item["_fusion_score"] += weight / (rank + 1)
                item["_best_rank"] = min(int(item.get("_best_rank", rank)), rank)
                owner_score = float(owner.get("score", 0) or 0)
                item["_best_owner_score"] = max(
                    float(item.get("_best_owner_score", 0) or 0),
                    owner_score,
                )
                if source not in item["sources"]:
                    item["sources"].append(source)
                if not item.get("name") and owner.get("name"):
                    item["name"] = owner.get("name")
                if item.get("mid") in (None, "") and owner.get("mid"):
                    item["mid"] = owner.get("mid")
                for field in (
                    "face",
                    "sample_title",
                    "sample_bvid",
                    "sample_pic",
                    "sample_view",
                ):
                    if item.get(field) in (None, "") and owner.get(field) not in (
                        None,
                        "",
                    ):
                        item[field] = owner.get(field)

        merged_items = list(merged.values())
        merged_items.sort(
            key=lambda item: (
                -float(item.get("_fusion_score", 0) or 0),
                -len(item.get("sources") or []),
                -float(item.get("_best_owner_score", 0) or 0),
                int(item.get("_best_rank", 999) or 999),
                str(item.get("name", "")),
            )
        )
        for item in merged_items:
            item["score"] = round(float(item.get("_fusion_score", 0) or 0), 4)
            item.pop("_fusion_score", None)
            item.pop("_best_rank", None)
            item.pop("_best_owner_score", None)
        return merged_items

    def _search_owners(self, args: dict) -> dict:
        resolved_args = _coerce_owner_search_args(args)
        text = str(resolved_args.get("text", "")).strip()
        bvids = _normalize_seed_values(resolved_args.get("bvids"))
        mids = _normalize_seed_values(resolved_args.get("mids"))
        requested_mode = str(resolved_args.get("mode", "auto") or "auto")
        relation_requested = requested_mode == "relation" or (
            not text and (bvids or mids)
        )

        if relation_requested and bvids:
            if not self._supports_relation_endpoint("related_owners_by_videos"):
                return {
                    "error": "Owner discovery unavailable",
                    "bvids": bvids,
                    "owners": [],
                }
            result = self.search_client.related_owners_by_videos(
                bvids=bvids,
                size=int(resolved_args.get("size", 10) or 10),
            )
            payload = self._format_relation_owner_result(result, "bvids", bvids)
            payload["mode"] = "relation"
            return payload

        if relation_requested and mids:
            if not self._supports_relation_endpoint("related_owners_by_owners"):
                return {
                    "error": "Owner discovery unavailable",
                    "mids": mids,
                    "owners": [],
                }
            result = self.search_client.related_owners_by_owners(
                mids=mids,
                size=int(resolved_args.get("size", 10) or 10),
            )
            payload = self._format_relation_owner_result(result, "mids", mids)
            payload["mode"] = "relation"
            return payload

        if not text:
            return {"error": "Missing text parameter", "owners": []}
        size = int(resolved_args.get("size", 8) or 8)
        max_owner_hits = max(self.max_results, size)
        google_query = f"{text} site:space.bilibili.com"

        source_specs: dict[str, object] = {
            "name": lambda: self.search_client.search_owners(
                text=text,
                mode="name",
                size=size,
            ),
            "topic": lambda: self.search_client.search_owners(
                text=text,
                mode="topic",
                size=size,
            ),
            "relation": lambda: self.search_client.search_owners(
                text=text,
                mode="relation",
                size=size,
            ),
        }
        if self._supports_relation_endpoint("related_owners_by_tokens"):
            source_specs["related_tokens"] = (
                lambda: self.search_client.related_owners_by_tokens(
                    text=text,
                    size=size,
                )
            )
        if self._is_google_available():
            source_specs["google_space"] = lambda: self.google_client.search(
                query=google_query,
                num=max(5, min(size, 10)),
                lang="zh-CN",
            )

        source_groups: list[dict] = []
        merged_sources: list[tuple[str, list[dict]]] = []
        google_results: list[dict] = []
        errors: list[str] = []

        max_workers = min(max(len(source_specs), 1), 5)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                source: executor.submit(fetcher)
                for source, fetcher in source_specs.items()
            }
            for source, future in futures.items():
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"error": str(exc)}
                if not isinstance(result, dict):
                    result = {}
                if source == "google_space":
                    if result.get("error"):
                        errors.append(str(result["error"]))
                        source_groups.append(
                            self._owner_source_group(
                                source=source,
                                owners=[],
                                max_hits=max_owner_hits,
                                text=text,
                                error=str(result["error"]),
                                query=google_query,
                                google_results=[],
                            )
                        )
                        continue
                    google_results = format_google_results(
                        result.get("results", []),
                        max_hits=max(5, size),
                    )
                    google_owners = []
                    for item in google_results:
                        if item.get("site_kind") != "space" or not item.get("mid"):
                            continue
                        google_owners.append(
                            {
                                "mid": item.get("mid"),
                                "name": _extract_google_space_candidate_name(
                                    item.get("title", "")
                                )
                                or f"B站用户{item.get('mid')}",
                                "score": result.get("result_count", 0),
                                "sample_title": item.get("title", ""),
                                "sources": [source],
                            }
                        )
                    source_groups.append(
                        self._owner_source_group(
                            source=source,
                            owners=google_owners,
                            max_hits=max_owner_hits,
                            text=text,
                            query=google_query,
                            google_results=google_results,
                        )
                    )
                    merged_sources.append((source, google_owners))
                    continue

                if result.get("error"):
                    errors.append(str(result["error"]))
                    source_groups.append(
                        self._owner_source_group(
                            source=source,
                            owners=[],
                            max_hits=max_owner_hits,
                            text=text,
                            error=str(result["error"]),
                        )
                    )
                    continue

                owners = list(result.get("owners") or [])
                source_groups.append(
                    self._owner_source_group(
                        source=source,
                        owners=owners,
                        max_hits=max_owner_hits,
                        text=text,
                    )
                )
                merged_sources.append((source, owners))

        owners = self._merge_owner_source_groups(merged_sources)
        payload = {
            "text": text,
            "mode": "aggregate",
            "requested_mode": requested_mode,
            "total_owners": len(owners),
            "owners": format_related_owners(owners, max_hits=max_owner_hits),
            "source_counts": {
                group["source"]: int(group.get("total_owners", 0) or 0)
                for group in source_groups
            },
            "source_groups": source_groups,
        }
        if google_results:
            payload["google_results"] = google_results[: max(5, size)]
            payload["google_query"] = google_query
        if errors and not owners and not google_results:
            payload["error"] = errors[0]
        return payload
