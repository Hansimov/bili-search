from __future__ import annotations

from sedb.elastic import ElasticOperator
from tclogger import logger

from configs.envs import ELASTIC_PRO_ENVS, SECRETS
from elastics.relations.tokens import sanitize_related_token_result
from elastics.videos.constants import SEARCH_MATCH_FIELDS, SUGGEST_MATCH_FIELDS


DEFAULT_RELATED_TOKEN_FIELDS = list(SUGGEST_MATCH_FIELDS)
DEFAULT_RELATED_OWNER_FIELDS = list(SEARCH_MATCH_FIELDS)
ES_COMPAT_MEDIA_TYPE = "application/vnd.elasticsearch+json; compatible-with=9"
RELATED_ENDPOINTS = [
    "related_tokens_by_tokens",
    "related_owners_by_tokens",
    "related_videos_by_videos",
    "related_owners_by_videos",
    "related_videos_by_owners",
    "related_owners_by_owners",
]


class RelationsClient:
    def __init__(
        self,
        index_name: str,
        elastic_env_name: str | None = None,
    ):
        self.index_name = index_name
        self.elastic_env_name = elastic_env_name
        self.init_es()

    def init_es(self):
        if self.elastic_env_name:
            elastic_envs = SECRETS[self.elastic_env_name]
        else:
            elastic_envs = ELASTIC_PRO_ENVS
        self.es = ElasticOperator(elastic_envs, connect_cls=self.__class__)

    @staticmethod
    def _normalize_fields(fields, default_fields: list[str]) -> list[str]:
        if fields is None:
            return list(default_fields)
        if isinstance(fields, str):
            values = [field.strip() for field in fields.split(",") if field.strip()]
            return values or list(default_fields)
        return [str(field).strip() for field in fields if str(field).strip()]

    @staticmethod
    def _normalize_seed_values(single_value, multi_values) -> list:
        if multi_values:
            return [value for value in multi_values if value not in [None, ""]]
        if single_value in [None, ""]:
            return []
        return [single_value]

    def _request(self, endpoint: str, payload: dict) -> dict:
        path = f"/{self.index_name}/_es_tok/{endpoint}"
        try:
            response = self.es.client.perform_request(
                method="POST",
                path=path,
                body=payload,
                headers={
                    "Accept": ES_COMPAT_MEDIA_TYPE,
                    "Content-Type": ES_COMPAT_MEDIA_TYPE,
                },
            )
            if hasattr(response, "body"):
                response = response.body
            if isinstance(response, dict):
                return response
            return dict(response)
        except Exception as exc:
            logger.warn(f"× Relation API error [{endpoint}]: {exc}")
            return {
                "error": str(exc),
                "relation": endpoint,
                **payload,
            }

    def related_tokens_by_tokens(
        self,
        text: str,
        *,
        fields=None,
        mode: str = "auto",
        size: int = 8,
        scan_limit: int = 128,
        use_pinyin: bool = True,
        **kwargs,
    ) -> dict:
        payload = {
            "text": text,
            "mode": mode,
            "fields": self._normalize_fields(fields, DEFAULT_RELATED_TOKEN_FIELDS),
            "size": size,
            "scan_limit": scan_limit,
            "use_pinyin": use_pinyin,
        }
        payload.update(
            {key: value for key, value in kwargs.items() if value is not None}
        )
        return sanitize_related_token_result(
            text,
            self._request("related_tokens_by_tokens", payload),
        )

    def related_owners_by_tokens(
        self,
        text: str,
        *,
        fields=None,
        size: int = 8,
        scan_limit: int = 128,
        use_pinyin: bool = True,
        **kwargs,
    ) -> dict:
        payload = {
            "text": text,
            "fields": self._normalize_fields(fields, DEFAULT_RELATED_OWNER_FIELDS),
            "size": size,
            "scan_limit": scan_limit,
            "use_pinyin": use_pinyin,
        }
        payload.update(
            {key: value for key, value in kwargs.items() if value is not None}
        )
        return self._request("related_owners_by_tokens", payload)

    def related_videos_by_videos(
        self,
        *,
        bvid: str | None = None,
        bvids: list[str] | None = None,
        size: int = 10,
        scan_limit: int = 128,
    ) -> dict:
        normalized_bvids = self._normalize_seed_values(bvid, bvids)
        payload = {
            "bvids": normalized_bvids,
            "size": size,
            "scan_limit": scan_limit,
        }
        return self._request("related_videos_by_videos", payload)

    def related_owners_by_videos(
        self,
        *,
        bvid: str | None = None,
        bvids: list[str] | None = None,
        size: int = 10,
        scan_limit: int = 128,
    ) -> dict:
        normalized_bvids = self._normalize_seed_values(bvid, bvids)
        payload = {
            "bvids": normalized_bvids,
            "size": size,
            "scan_limit": scan_limit,
        }
        return self._request("related_owners_by_videos", payload)

    def related_videos_by_owners(
        self,
        *,
        mid: int | None = None,
        mids: list[int] | None = None,
        size: int = 10,
        scan_limit: int = 128,
    ) -> dict:
        normalized_mids = self._normalize_seed_values(mid, mids)
        payload = {
            "mids": normalized_mids,
            "size": size,
            "scan_limit": scan_limit,
        }
        return self._request("related_videos_by_owners", payload)

    def related_owners_by_owners(
        self,
        *,
        mid: int | None = None,
        mids: list[int] | None = None,
        size: int = 10,
        scan_limit: int = 128,
    ) -> dict:
        normalized_mids = self._normalize_seed_values(mid, mids)
        payload = {
            "mids": normalized_mids,
            "size": size,
            "scan_limit": scan_limit,
        }
        return self._request("related_owners_by_owners", payload)
