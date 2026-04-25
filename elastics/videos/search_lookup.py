from copy import deepcopy
import re
from tclogger import get_now
from typing import Union

from elastics.videos.constants import SEARCH_LIMIT, SOURCE_FIELDS


class VideoSearchLookupMixin:
    def get_user_briefs(self, mids: list[Union[str, int]]) -> list[dict]:
        normalized_mids: list[int] = []
        seen_mids: set[int] = set()
        for mid in mids or []:
            try:
                normalized_mid = int(str(mid).strip())
            except (TypeError, ValueError):
                continue
            if normalized_mid in seen_mids:
                continue
            seen_mids.add(normalized_mid)
            normalized_mids.append(normalized_mid)

        if not normalized_mids:
            return []

        pipeline = [
            {"$match": {"mid": {"$in": normalized_mids}}},
            {
                "$project": {
                    "_id": 0,
                    "mid": 1,
                    "name": 1,
                    "face": 1,
                    "video_count": {"$size": {"$ifNull": ["$videos", []]}},
                }
            },
        ]
        cursor = self.mongo.get_agg_cursor(
            "users",
            pipeline,
            batch_size=min(max(len(normalized_mids), 8), 256),
        )
        docs = list(cursor)
        docs_by_mid = {
            int(doc.get("mid")): {
                "mid": int(doc.get("mid")),
                "name": str(doc.get("name") or "").strip(),
                "face": str(doc.get("face") or "").strip(),
                "video_count": int(doc.get("video_count") or 0),
            }
            for doc in docs
            if doc.get("mid") not in (None, "")
        }
        return [docs_by_mid[mid] for mid in normalized_mids if mid in docs_by_mid]

    @staticmethod
    def _normalize_lookup_bvid_key(bvid: Union[str, int]) -> str:
        return str(bvid or "").strip().upper()

    @staticmethod
    def _normalize_lookup_bvids(bvids: list[Union[str, int]] | None) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for bvid in bvids or []:
            text = str(bvid or "").strip()
            text_key = VideoSearchLookupMixin._normalize_lookup_bvid_key(text)
            if not text_key or text_key in seen:
                continue
            seen.add(text_key)
            normalized.append(text)
        return normalized

    @staticmethod
    def _normalize_lookup_mids(mids: list[Union[str, int]] | None) -> list[int]:
        normalized: list[int] = []
        seen: set[int] = set()
        for mid in mids or []:
            try:
                normalized_mid = int(str(mid).strip())
            except (TypeError, ValueError):
                continue
            if normalized_mid in seen:
                continue
            seen.add(normalized_mid)
            normalized.append(normalized_mid)
        return normalized

    @staticmethod
    def _lookup_is_missing(value) -> bool:
        if value is None:
            return True
        if value == "":
            return True
        if isinstance(value, (list, tuple, set, dict)) and not value:
            return True
        return False

    @classmethod
    def _merge_lookup_video_docs(
        cls, primary: dict | None, fallback: dict | None
    ) -> dict:
        if not primary and not fallback:
            return {}
        if not primary:
            return deepcopy(fallback or {})
        if not fallback:
            return deepcopy(primary)

        merged = deepcopy(primary)
        for key, value in (fallback or {}).items():
            if key not in merged or cls._lookup_is_missing(merged.get(key)):
                merged[key] = deepcopy(value)
                continue
            if isinstance(merged.get(key), dict) and isinstance(value, dict):
                merged[key] = cls._merge_lookup_video_docs(merged[key], value)
        return merged

    @staticmethod
    def _project_lookup_video_fields() -> dict:
        return {
            "_id": 0,
            "bvid": 1,
            "title": 1,
            "desc": 1,
            "pic": 1,
            "duration": 1,
            "pubdate": 1,
            "insert_at": 1,
            "tags": 1,
            "owner.mid": 1,
            "owner.name": 1,
            "owner.face": 1,
            "stat.view": 1,
            "stat.coin": 1,
            "stat.like": 1,
            "stat.favorite": 1,
            "stat.reply": 1,
            "stat.share": 1,
            "stat.danmaku": 1,
        }

    @classmethod
    def _normalize_lookup_video_doc(cls, doc: dict) -> dict:
        owner = doc.get("owner") if isinstance(doc.get("owner"), dict) else {}
        owner_mid = owner.get("mid")
        try:
            owner_mid = int(str(owner_mid).strip())
        except (TypeError, ValueError):
            owner_mid = None

        stat = doc.get("stat") if isinstance(doc.get("stat"), dict) else {}
        tags = doc.get("tags")
        if isinstance(tags, list):
            tags = ",".join(str(tag).strip() for tag in tags if str(tag or "").strip())

        normalized = {
            "bvid": str(doc.get("bvid") or "").strip(),
            "title": str(doc.get("title") or "").strip(),
            "desc": str(doc.get("desc") or "").strip(),
            "pic": str(doc.get("pic") or "").strip(),
            "duration": doc.get("duration"),
            "pubdate": doc.get("pubdate"),
            "insert_at": doc.get("insert_at"),
            "tags": tags,
            "owner": {
                "mid": owner_mid,
                "name": str(owner.get("name") or "").strip(),
                "face": str(owner.get("face") or "").strip(),
            },
            "stat": {
                key: stat.get(key)
                for key in (
                    "view",
                    "coin",
                    "like",
                    "favorite",
                    "reply",
                    "share",
                    "danmaku",
                )
                if stat.get(key) is not None
            },
        }
        return {
            key: value
            for key, value in normalized.items()
            if not cls._lookup_is_missing(value)
        }

    @staticmethod
    def _date_window_lower_bound(date_window: str | None) -> int | None:
        window = str(date_window or "").strip().lower()
        if not window:
            return None
        match = re.fullmatch(r"(\d+)\s*([dwmy])", window)
        if not match:
            return None
        value = int(match.group(1))
        unit = match.group(2)
        unit_seconds = {
            "d": 86400,
            "w": 86400 * 7,
            "m": 86400 * 30,
            "y": 86400 * 365,
        }
        return int(get_now().timestamp()) - value * unit_seconds[unit]

    def get_video_docs_by_bvids_from_mongo(
        self,
        bvids: list[Union[str, int]],
        verbose: bool = False,
    ) -> list[dict]:
        normalized_bvids = self._normalize_lookup_bvids(bvids)
        if not normalized_bvids:
            return []

        pipeline = [
            {"$match": {"bvid": {"$in": normalized_bvids}}},
            {"$project": self._project_lookup_video_fields()},
        ]
        cursor = self.mongo.get_agg_cursor(
            "videos",
            pipeline,
            batch_size=min(max(len(normalized_bvids), 8), 256),
        )
        docs = [self._normalize_lookup_video_doc(doc) for doc in cursor]
        docs_by_bvid = {
            self._normalize_lookup_bvid_key(doc.get("bvid")): doc
            for doc in docs
            if doc.get("bvid")
        }
        return [
            docs_by_bvid[self._normalize_lookup_bvid_key(bvid)]
            for bvid in normalized_bvids
            if self._normalize_lookup_bvid_key(bvid) in docs_by_bvid
        ]

    def get_recent_video_docs_by_mids_from_mongo(
        self,
        mids: list[Union[str, int]],
        *,
        limit: int = 10,
        date_window: str | None = None,
        exclude_bvids: list[Union[str, int]] | None = None,
        verbose: bool = False,
    ) -> list[dict]:
        normalized_mids = self._normalize_lookup_mids(mids)
        if not normalized_mids:
            return []

        normalized_excludes = set(self._normalize_lookup_bvids(exclude_bvids))
        match_clause: dict = {"owner.mid": {"$in": normalized_mids}}
        lower_bound = self._date_window_lower_bound(date_window)
        if lower_bound is not None:
            match_clause["pubdate"] = {"$gte": lower_bound}
        if normalized_excludes:
            match_clause["bvid"] = {"$nin": list(normalized_excludes)}

        pipeline = [
            {"$match": match_clause},
            {"$sort": {"pubdate": -1}},
            {"$limit": max(int(limit or 10), 1)},
            {"$project": self._project_lookup_video_fields()},
        ]
        cursor = self.mongo.get_agg_cursor(
            "videos",
            pipeline,
            batch_size=min(max(int(limit or 10), 8), 256),
        )
        return [self._normalize_lookup_video_doc(doc) for doc in cursor]

    def lookup_videos(
        self,
        *,
        bvids: list[Union[str, int]] | None = None,
        mids: list[Union[str, int]] | None = None,
        limit: int = 10,
        date_window: str | None = None,
        exclude_bvids: list[Union[str, int]] | None = None,
        verbose: bool = False,
    ) -> dict:
        normalized_bvids = self._normalize_lookup_bvids(bvids)
        normalized_mids = self._normalize_lookup_mids(mids)
        effective_limit = max(int(limit or 10), 1)

        if normalized_bvids:
            es_result = self.fetch_docs_by_bvids(
                normalized_bvids,
                source_fields=SOURCE_FIELDS,
                limit=len(normalized_bvids),
                verbose=verbose,
            )
            es_hits = es_result.get("hits") or []
            es_hits_by_bvid = {
                self._normalize_lookup_bvid_key(hit.get("bvid")): hit
                for hit in es_hits
                if hit.get("bvid")
            }
            mongo_hits = self.get_video_docs_by_bvids_from_mongo(
                normalized_bvids,
                verbose=verbose,
            )
            mongo_hits_by_bvid = {
                self._normalize_lookup_bvid_key(hit.get("bvid")): hit
                for hit in mongo_hits
                if hit.get("bvid")
            }
            merged_hits = []
            for bvid in normalized_bvids:
                bvid_key = self._normalize_lookup_bvid_key(bvid)
                merged_hit = self._merge_lookup_video_docs(
                    mongo_hits_by_bvid.get(bvid_key),
                    es_hits_by_bvid.get(bvid_key),
                )
                if merged_hit:
                    merged_hits.append(merged_hit)
            return {
                "lookup_by": "bvids",
                "bvids": normalized_bvids,
                "hits": merged_hits[:effective_limit],
                "total_hits": len(merged_hits),
                "source_counts": {
                    "mongo": len(mongo_hits_by_bvid),
                    "es": len(es_hits_by_bvid),
                },
            }

        if not normalized_mids:
            return {
                "lookup_by": "unknown",
                "hits": [],
                "total_hits": 0,
                "source_counts": {"mongo": 0, "es": 0},
            }

        mongo_hits = self.get_recent_video_docs_by_mids_from_mongo(
            normalized_mids,
            limit=effective_limit,
            date_window=date_window,
            exclude_bvids=exclude_bvids,
            verbose=verbose,
        )
        es_hits_by_bvid = {}
        if mongo_hits:
            es_result = self.fetch_docs_by_bvids(
                [hit.get("bvid") for hit in mongo_hits if hit.get("bvid")],
                source_fields=SOURCE_FIELDS,
                limit=len(mongo_hits),
                verbose=verbose,
            )
            es_hits_by_bvid = {
                self._normalize_lookup_bvid_key(hit.get("bvid")): hit
                for hit in (es_result.get("hits") or [])
                if hit.get("bvid")
            }

        merged_hits = []
        for hit in mongo_hits:
            bvid = self._normalize_lookup_bvid_key(hit.get("bvid"))
            merged_hits.append(
                self._merge_lookup_video_docs(hit, es_hits_by_bvid.get(bvid))
            )

        return {
            "lookup_by": "mids",
            "mids": normalized_mids,
            "date_window": str(date_window or "").strip(),
            "hits": merged_hits[:effective_limit],
            "total_hits": len(merged_hits),
            "source_counts": {
                "mongo": len(mongo_hits),
                "es": len(es_hits_by_bvid),
            },
        }
