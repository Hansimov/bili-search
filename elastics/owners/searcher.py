from __future__ import annotations

import re

from dataclasses import dataclass, replace
from sedb.elastic import ElasticOperator

from configs.envs import ELASTIC_PRO_ENVS, SECRETS
from elastics.relations import RelationsClient


OWNER_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+|[\u4e00-\u9fff]+")
OWNER_INTENT_PART_RE = re.compile(r"[A-Za-z]+|\d+|[\u4e00-\u9fff]+")
OWNER_INTENT_TOP_PROMOTION_MARGIN = 40.0
OWNER_INTENT_MIN_MATCH_SCORE = 6.0
OWNER_NAME_CANDIDATE_POOL_MIN = 16
OWNER_TOPIC_CANDIDATE_POOL_MIN = 10
OWNER_RELATION_CANDIDATE_POOL_MIN = 10


@dataclass(slots=True)
class OwnerCandidate:
    mid: int
    name: str
    score: float
    sources: list[str]
    face: str = ""
    sample_title: str = ""
    sample_bvid: str = ""
    sample_pic: str = ""
    sample_view: int | None = None

    def to_dict(self) -> dict:
        payload = {
            "mid": self.mid,
            "name": self.name,
            "score": round(float(self.score), 4),
            "sources": list(self.sources),
        }
        if self.face:
            payload["face"] = self.face
        if self.sample_title:
            payload["sample_title"] = self.sample_title
        if self.sample_bvid:
            payload["sample_bvid"] = self.sample_bvid
        if self.sample_pic:
            payload["sample_pic"] = self.sample_pic
        if self.sample_view is not None:
            payload["sample_view"] = self.sample_view
        return payload


class OwnerSearcher:
    def __init__(
        self,
        index_name: str,
        elastic_env_name: str | None = None,
        relations_client: RelationsClient | None = None,
    ):
        self.index_name = index_name
        self.elastic_env_name = elastic_env_name
        self.relations_client = relations_client or RelationsClient(
            index_name=index_name,
            elastic_env_name=elastic_env_name,
        )
        self.init_es()

    def init_es(self):
        if self.elastic_env_name:
            elastic_envs = SECRETS[self.elastic_env_name]
        else:
            elastic_envs = ELASTIC_PRO_ENVS
        self.es = ElasticOperator(elastic_envs, connect_cls=self.__class__)

    def search(
        self,
        text: str,
        *,
        size: int = 8,
        mode: str = "auto",
    ) -> dict:
        query = (text or "").strip()
        if not query:
            return {"text": "", "mode": mode, "owners": []}

        resolved_mode = str(mode or "auto").strip().lower()
        if resolved_mode not in {"auto", "name", "topic", "relation"}:
            resolved_mode = "auto"
        prepared_query = self._prepare_query(query, resolved_mode)

        name_hits: list[dict] = []
        if resolved_mode in {"auto", "name", "relation"}:
            name_hits = self._search_name_candidates(
                prepared_query,
                size=max(size * 2, OWNER_NAME_CANDIDATE_POOL_MIN),
            )

        topic_hits: list[dict] = []
        relation_hits: list[dict] = []
        should_probe_topic_hits = resolved_mode in {"auto", "topic"} or (
            resolved_mode == "name"
            and self._should_enrich_name_search_with_topic(
                prepared_query,
                name_hits,
            )
        )
        if should_probe_topic_hits:
            topic_hits = self._search_topic_candidates(
                prepared_query,
                size=max(size * 2, OWNER_TOPIC_CANDIDATE_POOL_MIN),
            )

        if resolved_mode == "relation":
            relation_hits = self._search_relation_candidates(
                prepared_query,
                name_hits=name_hits,
                size=max(size * 2, OWNER_RELATION_CANDIDATE_POOL_MIN),
            )
            if not relation_hits:
                topic_hits = self._search_topic_candidates(
                    prepared_query,
                    size=max(size * 2, OWNER_TOPIC_CANDIDATE_POOL_MIN),
                )

        merged = self._merge_candidates(
            query,
            name_hits=name_hits,
            topic_hits=topic_hits,
            relation_hits=relation_hits,
            size=size,
            mode=resolved_mode,
        )
        return {
            "text": query,
            "mode": resolved_mode,
            "total_owners": len(merged),
            "owners": [candidate.to_dict() for candidate in merged],
        }

    def _search_name_candidates(self, query: str, size: int) -> list[dict]:
        body = self._build_name_search_body(query=query, size=size)
        try:
            response = self.es.client.search(index=self.index_name, body=body)
            payload = response.body if hasattr(response, "body") else response
        except Exception:
            return []

        hits = ((payload or {}).get("hits") or {}).get("hits") or []
        results = []
        for rank, hit in enumerate(hits):
            source = hit.get("_source") or {}
            owner = source.get("owner") or {}
            mid = owner.get("mid")
            name = (owner.get("name") or "").strip()
            if not mid or not name:
                continue
            score = self._name_candidate_score(
                query, name, hit.get("_score") or 0.0, rank
            )
            results.append(
                {
                    "mid": int(mid),
                    "name": name,
                    "score": score,
                    "face": owner.get("face", ""),
                    "sample_title": source.get("title", ""),
                    "sample_bvid": source.get("bvid", ""),
                    "sample_pic": source.get("pic", ""),
                    "sample_view": ((source.get("stat") or {}).get("view")),
                    "sources": ["name"],
                }
            )
        return results

    def _load_owner_metadata(self, mids: list[int]) -> dict[int, dict]:
        normalized_mids = sorted({int(mid) for mid in mids if mid})
        if not normalized_mids:
            return {}

        body = {
            "_source": ["bvid", "title", "pic", "owner", "stat.view", "insert_at"],
            "size": min(max(len(normalized_mids), 8), 64),
            "query": {
                "terms": {
                    "owner.mid": normalized_mids,
                }
            },
            "collapse": {"field": "owner.mid"},
            "sort": [
                {"stat.view": {"order": "desc", "missing": "_last"}},
                {"insert_at": {"order": "desc", "missing": "_last"}},
            ],
        }

        try:
            response = self.es.client.search(index=self.index_name, body=body)
            payload = response.body if hasattr(response, "body") else response
        except Exception:
            return {}

        hits = ((payload or {}).get("hits") or {}).get("hits") or []
        metadata: dict[int, dict] = {}
        for hit in hits:
            source = hit.get("_source") or {}
            owner = source.get("owner") or {}
            mid = owner.get("mid")
            if not mid:
                continue
            metadata[int(mid)] = {
                "face": owner.get("face", ""),
                "sample_title": source.get("title", ""),
                "sample_bvid": source.get("bvid", ""),
                "sample_pic": source.get("pic", ""),
                "sample_view": ((source.get("stat") or {}).get("view")),
            }
        return metadata

    def _prepare_query(self, query: str, mode: str) -> str:
        prepared = (query or "").strip()
        prepared = re.sub(r"[?？,，。.!！:：]+", " ", prepared)
        prepared = re.sub(r"\s+", " ", prepared).strip()
        return prepared or query

    def _build_name_search_body(self, query: str, size: int) -> dict:
        wildcard_value = self._build_ordered_wildcard(query)
        should = [
            {"term": {"owner.name.keyword": {"value": query, "boost": 18}}},
            {"match_phrase": {"owner.name.words": {"query": query, "boost": 10}}},
            {
                "match_phrase_prefix": {
                    "owner.name.words": {
                        "query": query,
                        "max_expansions": 16,
                        "boost": 8,
                    }
                }
            },
            {
                "multi_match": {
                    "query": query,
                    "type": "cross_fields",
                    "operator": "and",
                    "fields": [
                        "owner.name.words^8",
                        "owner.name.suggest^6",
                    ],
                    "boost": 7,
                }
            },
            {
                "match": {
                    "owner.name.suggest": {
                        "query": query,
                        "operator": "and",
                        "boost": 5,
                    }
                }
            },
            {
                "match": {
                    "owner.name.words": {
                        "query": query,
                        "operator": "or",
                        "boost": 2,
                    }
                }
            },
        ]
        if wildcard_value:
            should.append(
                {
                    "wildcard": {
                        "owner.name.keyword": {
                            "value": wildcard_value,
                            "case_insensitive": True,
                            "boost": 12,
                        }
                    }
                }
            )

        return {
            "_source": ["bvid", "title", "owner", "stat.view", "insert_at"],
            "size": min(max(size, 8), 48),
            "query": {
                "bool": {
                    "should": should,
                    "minimum_should_match": 1,
                }
            },
            "collapse": {"field": "owner.mid"},
            "sort": [
                "_score",
                {"stat.view": {"order": "desc", "missing": "_last"}},
                {"insert_at": {"order": "desc", "missing": "_last"}},
            ],
        }

    def _build_ordered_wildcard(self, query: str) -> str | None:
        tokens = [
            token.lower() for token in OWNER_TOKEN_RE.findall(query) if token.strip()
        ]
        if not tokens:
            return None
        if len("".join(tokens)) > 24:
            return None
        return "*" + "*".join(tokens) + "*"

    def _name_candidate_score(
        self, query: str, owner_name: str, es_score: float, rank: int
    ) -> float:
        normalized_query = self._normalize_name(query)
        normalized_name = self._normalize_name(owner_name)
        query_tokens = self._owner_tokens(normalized_query)
        contains_all_tokens = bool(query_tokens) and all(
            token in normalized_name for token in query_tokens
        )
        ordered_tokens = bool(query_tokens) and self._ordered_token_match(
            normalized_name,
            query_tokens,
        )
        exact = normalized_query == normalized_name
        startswith = bool(normalized_query) and normalized_name.startswith(
            normalized_query
        )
        substring = bool(normalized_query) and normalized_query in normalized_name
        score = 120.0 - (rank * 3.5)
        score += min(float(es_score), 30.0)
        if exact:
            score += 40.0
        elif startswith:
            score += 22.0
        elif substring:
            score += 14.0
        if contains_all_tokens:
            score += 24.0
        if ordered_tokens:
            score += 10.0
        return score

    def _search_topic_candidates(self, query: str, size: int) -> list[dict]:
        if self.relations_client is None:
            return []
        try:
            result = self.relations_client.related_owners_by_tokens(
                text=query,
                size=size,
            )
        except Exception:
            return []
        if result.get("error"):
            return []
        owners = result.get("owners") or []
        return [
            {
                "mid": int(owner.get("mid")),
                "name": (owner.get("name") or "").strip(),
                "score": 80.0
                - (index * 3.0)
                + min(float(owner.get("score") or 0.0), 20.0),
                "sources": ["topic"],
            }
            for index, owner in enumerate(owners)
            if owner.get("mid") and (owner.get("name") or "").strip()
        ]

    def _search_relation_candidates(
        self,
        query: str,
        *,
        name_hits: list[dict],
        size: int,
    ) -> list[dict]:
        if self.relations_client is None:
            return []
        seed_mids = [
            candidate.get("mid") for candidate in name_hits[:3] if candidate.get("mid")
        ]
        if not seed_mids:
            topic_hits = self._search_topic_candidates(query, size=max(size, 10))
            seed_mids = [
                candidate.get("mid")
                for candidate in topic_hits[:3]
                if candidate.get("mid")
            ]
        if not seed_mids:
            return []
        try:
            result = self.relations_client.related_owners_by_owners(
                mids=seed_mids,
                size=size,
            )
        except Exception:
            return []
        if result.get("error"):
            return []
        owners = result.get("owners") or []
        return [
            {
                "mid": int(owner.get("mid")),
                "name": (owner.get("name") or "").strip(),
                "score": 88.0
                - (index * 3.0)
                + min(float(owner.get("score") or 0.0), 18.0),
                "sources": ["relation"],
            }
            for index, owner in enumerate(owners)
            if owner.get("mid")
            and (owner.get("name") or "").strip()
            and int(owner.get("mid")) not in set(seed_mids)
        ]

    def _merge_candidates(
        self,
        query: str,
        *,
        name_hits: list[dict],
        topic_hits: list[dict],
        relation_hits: list[dict],
        size: int,
        mode: str,
    ) -> list[OwnerCandidate]:
        merged: dict[int, dict] = {}
        for bucket in [name_hits, topic_hits, relation_hits]:
            for candidate in bucket:
                mid = int(candidate["mid"])
                slot = merged.setdefault(
                    mid,
                    {
                        "mid": mid,
                        "name": candidate["name"],
                        "score": 0.0,
                        "sources": [],
                        "face": candidate.get("face", ""),
                        "sample_title": candidate.get("sample_title", ""),
                        "sample_bvid": candidate.get("sample_bvid", ""),
                        "sample_pic": candidate.get("sample_pic", ""),
                        "sample_view": candidate.get("sample_view"),
                    },
                )
                slot["score"] += float(candidate.get("score") or 0.0)
                for source in candidate.get("sources") or []:
                    if source not in slot["sources"]:
                        slot["sources"].append(source)
                if not slot.get("face") and candidate.get("face"):
                    slot["face"] = candidate["face"]
                if not slot.get("sample_title") and candidate.get("sample_title"):
                    slot["sample_title"] = candidate["sample_title"]
                    slot["sample_bvid"] = candidate.get("sample_bvid", "")
                    slot["sample_pic"] = candidate.get("sample_pic", "")
                    slot["sample_view"] = candidate.get("sample_view")

        metadata_by_mid = self._load_owner_metadata(list(merged.keys()))
        for mid, slot in merged.items():
            metadata = metadata_by_mid.get(mid) or {}
            if not slot.get("face") and metadata.get("face"):
                slot["face"] = metadata["face"]
            if not slot.get("sample_title") and metadata.get("sample_title"):
                slot["sample_title"] = metadata["sample_title"]
                slot["sample_bvid"] = metadata.get("sample_bvid", "")
                slot["sample_pic"] = metadata.get("sample_pic", "")
                slot["sample_view"] = metadata.get("sample_view")
            elif not slot.get("sample_pic") and metadata.get("sample_pic"):
                slot["sample_pic"] = metadata["sample_pic"]

        normalized_query = self._normalize_name(query)
        results: list[OwnerCandidate] = []
        for candidate in merged.values():
            normalized_name = self._normalize_name(candidate["name"])
            if (
                mode in {"auto", "name", "relation"}
                and normalized_query == normalized_name
            ):
                candidate["score"] += 40.0
            elif (
                mode in {"auto", "name", "relation"}
                and normalized_query
                and normalized_query in normalized_name
            ):
                candidate["score"] += 18.0
            if "name" in candidate["sources"] and "topic" in candidate["sources"]:
                candidate["score"] += 18.0
            if "name" in candidate["sources"] and "relation" in candidate["sources"]:
                candidate["score"] += 12.0
            results.append(
                OwnerCandidate(
                    mid=candidate["mid"],
                    name=candidate["name"],
                    score=candidate["score"],
                    sources=candidate["sources"],
                    face=candidate.get("face", ""),
                    sample_title=candidate.get("sample_title", ""),
                    sample_bvid=candidate.get("sample_bvid", ""),
                    sample_pic=candidate.get("sample_pic", ""),
                    sample_view=candidate.get("sample_view"),
                )
            )

        results = self._maybe_promote_owner_intent_candidate(
            query,
            results,
            mode=mode,
        )
        results.sort(key=lambda item: (-item.score, item.name, item.mid))
        return results[:size]

    def _should_enrich_name_search_with_topic(
        self,
        query: str,
        name_hits: list[dict],
    ) -> bool:
        if not self._is_owner_intent_like_query(query):
            return False
        return not self._has_exact_name_hit(query, name_hits)

    def _has_exact_name_hit(self, query: str, name_hits: list[dict]) -> bool:
        normalized_query = self._normalize_name(query)
        if not normalized_query:
            return False
        for candidate in name_hits[:3]:
            normalized_name = self._normalize_name(candidate.get("name") or "")
            if normalized_name == normalized_query:
                return True
        return False

    def _maybe_promote_owner_intent_candidate(
        self,
        query: str,
        results: list[OwnerCandidate],
        *,
        mode: str,
    ) -> list[OwnerCandidate]:
        if mode not in {"auto", "name", "relation"}:
            return results
        if not results or not self._is_owner_intent_like_query(query):
            return results

        best_candidate: OwnerCandidate | None = None
        best_score = float("-inf")
        for candidate in results:
            if not self._candidate_has_owner_intent_source(candidate):
                continue
            score = self._owner_intent_match_score(query, candidate.name)
            score += self._owner_intent_candidate_bonus(candidate)
            score += min(candidate.score, 400.0) * 0.03
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None or best_score < OWNER_INTENT_MIN_MATCH_SCORE:
            return results

        top_score = max(candidate.score for candidate in results)
        promoted_score = max(
            best_candidate.score,
            top_score + OWNER_INTENT_TOP_PROMOTION_MARGIN,
        )
        return [
            (
                replace(candidate, score=promoted_score)
                if candidate.mid == best_candidate.mid
                else candidate
            )
            for candidate in results
        ]

    @staticmethod
    def _candidate_has_owner_intent_source(candidate: OwnerCandidate) -> bool:
        return any(source in {"topic", "relation"} for source in candidate.sources)

    @staticmethod
    def _owner_intent_candidate_bonus(candidate: OwnerCandidate) -> float:
        score = 0.0
        if "name" in candidate.sources:
            score += 1.6
        if len(candidate.sources) >= 2:
            score += 0.8
        return score

    def _owner_intent_match_score(self, query: str, owner_name: str) -> float:
        normalized_query = self._normalize_name(query)
        normalized_name = self._normalize_name(owner_name)
        if not normalized_query or not normalized_name:
            return 0.0

        score = 0.0
        if normalized_name == normalized_query:
            score += 8.0
        if normalized_name.startswith(normalized_query):
            score += 5.0
        if normalized_query in normalized_name:
            score += 3.0

        query_parts = self._owner_intent_parts(normalized_query)
        matched_parts = 0
        matched_chinese = False
        matched_digits = False
        search_from = 0
        for part in query_parts:
            index = normalized_name.find(part, search_from)
            if index < 0:
                continue
            matched_parts += 1
            search_from = index + len(part)
            if part.isdigit():
                matched_digits = True
                score += 2.2
            elif self._contains_chinese(part):
                matched_chinese = True
                score += 2.6 if index == 0 else 1.8
            else:
                score += 1.1

        if query_parts and matched_parts == len(query_parts):
            score += 3.5
        if matched_chinese and matched_digits:
            score += 2.4

        for part in self._owner_intent_tail_parts(normalized_query):
            if part not in normalized_name:
                continue
            if len(part) >= 3:
                score += 4.2
            else:
                score += 1.4
            if normalized_name.endswith(part):
                score += 0.8

        longest_common = self._best_common_substring_length(
            normalized_query,
            normalized_name,
        )
        if longest_common >= 2:
            score += float(longest_common) * 1.1

        return score

    def _owner_intent_parts(self, normalized_query: str) -> list[str]:
        parts = [
            part
            for part in OWNER_INTENT_PART_RE.findall(normalized_query)
            if part.strip()
        ]
        if len(parts) == 1 and self._contains_chinese(parts[0]) and len(parts[0]) >= 6:
            chinese_part = parts[0]
            for size in (3, 2):
                suffix = chinese_part[-size:]
                if suffix and suffix not in parts:
                    parts.append(suffix)
        return parts

    def _owner_intent_tail_parts(self, normalized_query: str) -> list[str]:
        if not self._contains_chinese(normalized_query) or len(normalized_query) < 6:
            return []

        parts: list[str] = []
        for size in (3, 2):
            suffix = normalized_query[-size:]
            if suffix and suffix not in parts:
                parts.append(suffix)
        return parts

    @staticmethod
    def _is_owner_intent_like_query(text: str) -> bool:
        normalized = (text or "").strip()
        if not normalized or any(char.isspace() for char in normalized):
            return False
        codepoint_length = len(normalized)
        return 2 <= codepoint_length <= 24

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    @staticmethod
    def _best_common_substring_length(left: str, right: str) -> int:
        if not left or not right:
            return 0
        best = 0
        matrix = [0] * (len(right) + 1)
        for left_char in left:
            next_row = [0] * (len(right) + 1)
            for index, right_char in enumerate(right, start=1):
                if left_char != right_char:
                    continue
                next_row[index] = matrix[index - 1] + 1
                if next_row[index] > best:
                    best = next_row[index]
            matrix = next_row
        return best

    @staticmethod
    def _owner_tokens(text: str) -> list[str]:
        return [
            token.lower() for token in OWNER_TOKEN_RE.findall(text) if token.strip()
        ]

    @staticmethod
    def _ordered_token_match(text: str, tokens: list[str]) -> bool:
        position = 0
        for token in tokens:
            index = text.find(token, position)
            if index < 0:
                return False
            position = index + len(token)
        return True

    @staticmethod
    def _normalize_name(value: str) -> str:
        return re.sub(r"\s+", "", (value or "").strip()).lower()
