from __future__ import annotations

import re

from llms.chat.planning import DEFAULT_TOOL_PLANNING_PLUGINS
from llms.chat.planning import ToolPlanningContext, apply_tool_planning_plugins


_RECENT_VIDEO_INTENT_RE = re.compile(
    r"最近|最新|近\d+[天日周月]|近期|这几天|刚发|刚更新|新发"
)
_VIDEO_SEARCH_INTENT_RE = re.compile(
    r"视频|播放|剧情解析|解说|教程|攻略|推荐几条|找几条|热门|高播放|代表作"
)
_USER_FILTER_RE = re.compile(r":user=([^\s]+)")
_TOKEN_OPTION_CANONICAL_RE = re.compile(r"[^A-Za-z0-9\u4e00-\u9fff]+")
_GOOGLE_KEYWORD_BOOTSTRAP_RE = re.compile(
    r"怎么搜|怎么写标题|怎么写题目|搜什么关键词|摸一下关键词|摸关键词|标题写法"
)
_GOOGLE_CREATOR_BOOTSTRAP_RE = re.compile(
    r"不知道作者叫什么|不知道up主叫什么|不知道UP主叫什么|先帮我摸几个up主|先帮我摸几个UP主|先帮我找几个up主|先帮我找几个UP主|先帮我摸几个作者|先帮我找几个作者|谁在做|有哪些up主|有哪些UP主|有哪些作者|做这类内容的up主|做这类内容的UP主"
)


class ToolPlanningMixin:
    @staticmethod
    def _cleanup_google_probe_fragment(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        cleaned = re.sub(r"^(?:但我|但是我)\s*", "", cleaned)
        cleaned = re.sub(r"\s*(?:但我|但是我)$", "", cleaned)
        cleaned = re.sub(r"\s*的$", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def _split_google_bootstrap_phrases(cls, text: str) -> list[str]:
        normalized = cls._normalize_entity_focused_query_text(text)
        normalized = cls._cleanup_google_probe_fragment(normalized)
        if not normalized:
            return []

        parts = re.split(r"\s*(?:和|与|以及|及|、|，|,|/|／|\+)\s*", normalized)
        phrases = [
            cls._cleanup_google_probe_fragment(part)
            for part in parts
            if cls._cleanup_google_probe_fragment(part)
        ]
        phrases = list(dict.fromkeys(phrases))
        return phrases or [normalized]

    @classmethod
    def _build_site_scoped_google_commands(
        cls,
        scope: str,
        *,
        combined_query: str,
        split_queries: list[str] | None = None,
    ) -> list[dict]:
        queries: list[str] = []

        def add_query(text: str) -> None:
            normalized = cls._normalize_entity_focused_query_text(text)
            normalized = cls._cleanup_google_probe_fragment(normalized)
            if not normalized:
                return
            final_query = f"{normalized} {scope}".strip()
            if final_query not in queries:
                queries.append(final_query)

        for text in split_queries or []:
            add_query(text)
        add_query(combined_query)

        return [
            {
                "type": "search_google",
                "args": {"query": query},
            }
            for query in queries
        ]

    @classmethod
    def _normalize_google_keyword_bootstrap_query(cls, text: str) -> str:
        query = str(text or "")
        query = re.sub(
            r"(?:B站|b站|哔哩哔哩|bilibili)", " ", query, flags=re.IGNORECASE
        )
        query = re.sub(
            r"(?:我想找|想找|我想看|想看|先帮我|帮我|给我|顺便|再|先|有没有|有吗|讲|里|上)",
            " ",
            query,
        )
        query = re.sub(
            r"(?:但我不确定大家会怎么写标题|我不确定大家会怎么写标题|大家会怎么写标题|大家怎么写标题|怎么写标题|怎么写题目|一般怎么搜|怎么搜|搜什么关键词|先帮我摸一下关键词|先摸一下关键词|摸一下关键词|摸关键词|标题写法|给我几条视频|给我几条|几条视频|专栏里有没有|专栏文章|文章|视频|内容)",
            " ",
            query,
        )
        query = re.sub(r"\b(?:的|但我|但是我)\b", " ", query)
        query = cls._normalize_entity_focused_query_text(query)
        query = re.sub(r"\s+", " ", query).strip()
        return query

    @classmethod
    def _build_google_keyword_bootstrap_commands(
        cls,
        commands: list[dict],
        messages: list[dict],
        last_tool_results: list[dict] | None,
    ) -> list[dict]:
        latest_user_text = cls._get_latest_user_text(messages)
        if not latest_user_text or not _GOOGLE_KEYWORD_BOOTSTRAP_RE.search(
            latest_user_text
        ):
            return commands
        if any(command.get("type") == "search_google" for command in commands or []):
            return commands
        if any(
            result_item.get("type") == "search_google"
            for result_item in last_tool_results or []
        ):
            return commands

        normalized_query = cls._normalize_google_keyword_bootstrap_query(
            latest_user_text
        )
        if not normalized_query:
            return commands

        split_queries = cls._split_google_bootstrap_phrases(normalized_query)
        combined_query = (
            " ".join(split_queries) if len(split_queries) > 1 else normalized_query
        )

        if re.search(r"专栏|文章|read", latest_user_text):
            scope = "site:bilibili.com/read"
        elif re.search(r"作者|UP主|up主|博主|账号|用户页|谁在做", latest_user_text):
            scope = "site:space.bilibili.com"
        elif cls._wants_video_results(messages, commands):
            scope = "site:bilibili.com/video"
        else:
            scope = "site:bilibili.com"

        google_commands = cls._build_site_scoped_google_commands(
            scope,
            combined_query=combined_query,
            split_queries=split_queries,
        )
        return google_commands + list(commands or [])

    @classmethod
    def _normalize_google_creator_bootstrap_query(cls, text: str) -> str:
        query = str(text or "")
        query = re.sub(
            r"(?:B站|b站|哔哩哔哩|bilibili)", " ", query, flags=re.IGNORECASE
        )
        query = re.sub(
            r"(?:我想找|想找|先帮我|帮我|给我|推荐|找一下|找找|看看|想看看|我想看|想看|有没有|有吗|哪些|几个|一些)",
            " ",
            query,
        )
        query = re.sub(
            r"(?:不知道作者叫什么|不知道up主叫什么|不知道UP主叫什么|先帮我摸几个up主|先帮我摸几个UP主|先帮我找几个up主|先帮我找几个UP主|先帮我摸几个作者|先帮我找几个作者|谁在做|有哪些up主|有哪些UP主|有哪些作者|UP主|up主|作者|博主|账号|创作者|做这类内容的|这类内容)",
            " ",
            query,
        )
        query = re.sub(r"\b(?:的|但我|但是我)\b", " ", query)
        query = cls._normalize_entity_focused_query_text(query)
        query = re.sub(r"\s+", " ", query).strip()
        query = re.sub(r"^(?:做)\s*", "", query)
        query = re.sub(r"\s*(?:的)$", "", query)
        query = re.sub(r"(?:但我|但是我|摸)$", "", query)
        query = re.sub(r"\s+", " ", query).strip()
        return query

    @classmethod
    def _build_google_creator_bootstrap_commands(
        cls,
        commands: list[dict],
        messages: list[dict],
        last_tool_results: list[dict] | None,
    ) -> list[dict]:
        latest_user_text = cls._get_latest_user_text(messages)
        if not latest_user_text or not _GOOGLE_CREATOR_BOOTSTRAP_RE.search(
            latest_user_text
        ):
            return commands
        if any(command.get("type") == "search_google" for command in commands or []):
            return commands
        if any(
            result_item.get("type") == "search_google"
            for result_item in last_tool_results or []
        ):
            return commands

        normalized_query = cls._normalize_google_creator_bootstrap_query(
            latest_user_text
        )
        if not normalized_query:
            return commands

        split_queries = cls._split_google_bootstrap_phrases(normalized_query)
        combined_query = (
            " ".join(split_queries) if len(split_queries) > 1 else normalized_query
        )

        google_commands = cls._build_site_scoped_google_commands(
            "site:space.bilibili.com",
            combined_query=combined_query,
            split_queries=split_queries,
        )
        return google_commands + list(commands or [])

    @classmethod
    def _collect_unresolved_owner_aliases(
        cls,
        results: list[dict] | None,
        hint_tokens: list[str] | None = None,
    ) -> list[str]:
        unresolved: list[str] = []
        for result_item in results or []:
            if result_item.get("type") != "search_owners":
                continue
            result = result_item.get("result") or {}
            source_text = str(
                result.get("text") or result_item.get("args", {}).get("text") or ""
            ).strip()
            owners = result.get("owners") or []
            candidate = cls._select_best_owner_candidate(
                source_text,
                owners,
                hint_tokens=hint_tokens,
            )
            if cls._is_confident_owner_candidate(
                source_text,
                candidate,
                hint_tokens=hint_tokens,
                owners=owners,
            ):
                continue
            if source_text and source_text not in unresolved:
                unresolved.append(source_text)
        return unresolved

    @classmethod
    def _should_google_owner_bootstrap_alias(cls, text: str) -> bool:
        normalized = cls._normalize_name_key(text)
        if not normalized:
            return False
        if cls._is_short_ambiguous_owner_text(text):
            return True
        return bool(re.search(r"\d", normalized)) and len(normalized) <= 6

    @classmethod
    def _build_google_owner_bootstrap_commands(
        cls,
        commands: list[dict],
        messages: list[dict],
        last_tool_results: list[dict] | None,
    ) -> list[dict]:
        latest_user_text = cls._get_latest_user_text(messages)
        if not latest_user_text or not cls._wants_video_results(messages, commands):
            return []
        if any(command.get("type") == "search_google" for command in commands or []):
            return []
        if any(
            result_item.get("type") == "search_google"
            for result_item in last_tool_results or []
        ):
            return []

        owner_results = [
            result_item
            for result_item in last_tool_results or []
            if result_item.get("type") == "search_owners"
        ]
        if not owner_results:
            return []

        hint_tokens = cls._collect_owner_context_hints(owner_results)
        unresolved_aliases = cls._collect_unresolved_owner_aliases(owner_results)
        bootstrap_aliases: list[str] = []
        primary_hint = hint_tokens[0] if hint_tokens else ""
        for result_item in owner_results:
            result = result_item.get("result") or {}
            source_text = str(
                result.get("text") or result_item.get("args", {}).get("text") or ""
            ).strip()
            owners = result.get("owners") or []
            if not cls._should_google_owner_bootstrap_alias(source_text):
                continue
            if source_text in unresolved_aliases:
                bootstrap_aliases.append(source_text)
                continue
            if not primary_hint or primary_hint in source_text:
                continue
            contextual_hints = cls._collect_near_top_owner_context_hints(
                source_text, owners
            )
            if primary_hint in contextual_hints:
                bootstrap_aliases.append(source_text)
        bootstrap_aliases = list(dict.fromkeys(bootstrap_aliases))
        if not bootstrap_aliases:
            return []

        query_terms: list[str] = []
        for token in hint_tokens[:2]:
            if token and token not in query_terms:
                query_terms.append(token)
        for alias in bootstrap_aliases:
            if alias not in query_terms:
                query_terms.append(alias)

        normalized_query = cls._normalize_entity_focused_query_text(
            " ".join(query_terms)
        )
        if not normalized_query:
            return []

        return cls._build_site_scoped_google_commands(
            "site:space.bilibili.com",
            combined_query=normalized_query,
            split_queries=bootstrap_aliases,
        )

    @staticmethod
    def _extract_google_space_candidate_name(title: str) -> str:
        name = str(title or "").strip()
        if not name:
            return ""
        name = re.sub(r"(?:的个人空间|个人空间|主页).*$", "", name)
        name = re.sub(r"(?:[-|｜].*)$", "", name)
        name = name.strip(" -_|｜·，。！？?：:[]()（）")
        return name

    @classmethod
    def _build_google_space_owner_followup_commands(
        cls,
        commands: list[dict],
        messages: list[dict],
        last_tool_results: list[dict] | None,
    ) -> list[dict]:
        latest_user_text = cls._get_latest_user_text(messages)
        if not latest_user_text:
            return commands
        if not re.search(
            r"作者|UP主|up主|博主|创作者|谁在做|摸几个作者|摸几个up主|摸几个UP主",
            latest_user_text,
        ):
            return commands
        if any(
            command.get("type") in {"search_owners", "related_owners_by_tokens"}
            for command in commands or []
        ):
            return commands

        candidate_names: list[str] = []
        saw_google_probe = False
        for result_item in last_tool_results or []:
            if result_item.get("type") != "search_google":
                continue
            saw_google_probe = True
            result = result_item.get("result") or {}
            for google_result in result.get("results") or []:
                if google_result.get("site_kind") != "space":
                    continue
                candidate_name = cls._extract_google_space_candidate_name(
                    google_result.get("title")
                )
                if not candidate_name:
                    continue
                candidate_key = cls._normalize_name_key(candidate_name)
                if not candidate_key:
                    continue
                if candidate_name not in candidate_names:
                    candidate_names.append(candidate_name)

        if not saw_google_probe:
            return commands

        if not candidate_names:
            normalized_topic = cls._normalize_google_creator_bootstrap_query(
                latest_user_text
            )
            if not normalized_topic:
                return commands
            owner_commands = [
                {
                    "type": "search_owners",
                    "args": {"text": normalized_topic, "mode": "topic"},
                }
            ]
            passthrough = [
                command
                for command in commands or []
                if command.get("type")
                not in {"search_videos", "related_tokens_by_tokens"}
            ]
            return passthrough + owner_commands

        owner_commands = [
            {"type": "search_owners", "args": {"text": name, "mode": "name"}}
            for name in candidate_names[:5]
        ]
        passthrough = [
            command
            for command in commands or []
            if command.get("type") not in {"search_videos", "related_tokens_by_tokens"}
        ]
        return passthrough + owner_commands

    @staticmethod
    def _extract_user_filters_from_query(query: str) -> list[str]:
        text = str(query or "")
        if ":uid=" in text:
            return []
        return [name.strip() for name in _USER_FILTER_RE.findall(text) if name.strip()]

    @classmethod
    def _extract_owner_candidates_from_commands(cls, commands: list[dict]) -> list[str]:
        names: list[str] = []
        for command in commands or []:
            if command.get("type") != "search_videos":
                continue
            args = command.get("args") or {}
            queries = args.get("queries")
            if isinstance(queries, str):
                queries = [queries]
            if not isinstance(queries, list):
                continue
            for query in queries:
                for name in cls._extract_user_filters_from_query(str(query)):
                    if name not in names:
                        names.append(name)
        return names

    @classmethod
    def _merge_owner_result_context(
        cls,
        context: dict[str, dict] | None,
        results: list[dict] | None,
    ) -> dict[str, dict]:
        merged = dict(context or {})
        for result_item in results or []:
            if result_item.get("type") != "search_owners":
                continue
            result = result_item.get("result") or {}
            source_text = str(
                result.get("text") or result_item.get("args", {}).get("text") or ""
            ).strip()
            key = cls._normalize_name_key(source_text)
            if key:
                merged[key] = result_item
        return merged

    @classmethod
    def _build_contextual_owner_search_commands(
        cls,
        messages: list[dict],
        last_tool_results: list[dict] | None,
    ) -> list[dict]:
        latest_user_text = cls._get_latest_user_text(messages)
        if not latest_user_text or not cls._wants_video_results(messages):
            return []

        hint_tokens = cls._collect_owner_context_hints(last_tool_results)
        if not hint_tokens:
            return []

        commands: list[dict] = []
        seen_texts: set[str] = {
            cls._normalize_name_key(
                str(
                    (result_item.get("result") or {}).get("text")
                    or (result_item.get("args") or {}).get("text")
                    or ""
                )
            )
            for result_item in last_tool_results or []
            if result_item.get("type") == "search_owners"
        }
        primary_hint = hint_tokens[0]
        for result_item in last_tool_results or []:
            if result_item.get("type") != "search_owners":
                continue
            result = result_item.get("result") or {}
            source_text = str(
                result.get("text") or result_item.get("args", {}).get("text") or ""
            ).strip()
            owners = result.get("owners") or []
            if not source_text or primary_hint in source_text:
                continue

            candidate = cls._select_best_owner_candidate(
                source_text,
                owners,
                hint_tokens=hint_tokens,
            )
            is_confident = cls._is_confident_owner_candidate(
                source_text,
                candidate,
                hint_tokens=hint_tokens,
                owners=owners,
            )
            should_retry = not is_confident
            if (
                not should_retry
                and cls._should_google_owner_bootstrap_alias(source_text)
                and primary_hint not in source_text
            ):
                contextual_hints = cls._collect_near_top_owner_context_hints(
                    source_text,
                    owners,
                )
                should_retry = primary_hint in contextual_hints

            if not should_retry:
                continue

            contextual_text = f"{primary_hint}{source_text}"
            contextual_key = cls._normalize_name_key(contextual_text)
            if contextual_key in seen_texts:
                continue
            seen_texts.add(contextual_key)
            commands.append(
                {
                    "type": "search_owners",
                    "args": {"text": contextual_text, "mode": "name"},
                }
            )

        return commands

    @classmethod
    def _has_unresolved_owner_results(cls, results: list[dict] | None) -> bool:
        for result_item in results or []:
            if result_item.get("type") != "search_owners":
                continue
            result = result_item.get("result") or {}
            source_text = str(
                result.get("text") or result_item.get("args", {}).get("text") or ""
            ).strip()
            owners = result.get("owners") or []
            candidate = cls._select_best_owner_candidate(source_text, owners)
            if not cls._is_confident_owner_candidate(
                source_text,
                candidate,
                owners=owners,
            ):
                return True
        return False

    @classmethod
    def _build_owner_assisted_video_search_commands(
        cls,
        commands: list[dict],
        messages: list[dict],
        last_tool_results: list[dict] | None,
    ) -> list[dict]:
        if not last_tool_results:
            return commands

        owner_only_commands = bool(commands) and all(
            command.get("type") == "search_owners" for command in commands
        )
        owner_result_stage = bool(last_tool_results) and all(
            result_item.get("type") == "search_owners"
            for result_item in last_tool_results or []
        )
        recent_video_stage = cls._wants_recent_video_results(messages, commands)

        latest_user_text = cls._get_latest_user_text(messages)
        if not latest_user_text or not cls._wants_video_results(messages, commands):
            return commands

        contextual_owner_commands = cls._build_contextual_owner_search_commands(
            messages,
            last_tool_results,
        )
        google_owner_commands = cls._build_google_owner_bootstrap_commands(
            commands,
            messages,
            last_tool_results,
        )
        if contextual_owner_commands and (
            not commands
            or owner_only_commands
            or cls._has_unresolved_owner_results(last_tool_results)
        ):
            return contextual_owner_commands

        if google_owner_commands and (
            not commands
            or owner_only_commands
            or cls._has_unresolved_owner_results(last_tool_results)
        ):
            return google_owner_commands

        if (
            commands
            and not owner_only_commands
            and not (owner_result_stage and recent_video_stage)
        ):
            return commands

        window = cls._extract_recent_window(latest_user_text)
        queries: list[str] = []
        effective_owner_results = cls._filter_superseded_owner_results(
            last_tool_results
        )
        for result_item in effective_owner_results:
            if result_item.get("type") != "search_owners":
                continue
            result = result_item.get("result") or {}
            source_text = str(
                result.get("text") or result_item.get("args", {}).get("text") or ""
            ).strip()
            owners = result.get("owners") or []
            candidate = cls._select_best_owner_candidate(source_text, owners)
            if not cls._is_confident_owner_candidate(
                source_text,
                candidate,
                owners=owners,
            ):
                return commands

            owner_mid = candidate.get("mid")
            owner_name = str(candidate.get("name") or "").strip()
            if owner_mid:
                query = f":uid={int(owner_mid)} :date<={window}"
            elif owner_name:
                query = f":user={owner_name} :date<={window}"
            else:
                return commands
            if query not in queries:
                queries.append(query)

        if not queries:
            return commands
        return [{"type": "search_videos", "args": {"queries": queries}}]

    @classmethod
    def _build_owner_resolution_commands(
        cls,
        commands: list[dict],
        last_tool_results: list[dict] | None,
    ) -> list[dict]:
        resolved_map = cls._extract_resolved_owner_map(last_tool_results)
        in_flight = {
            cls._normalize_name_key(str((command.get("args") or {}).get("text") or ""))
            for command in commands or []
            if command.get("type") == "search_owners"
            and str((command.get("args") or {}).get("mode") or "auto").lower()
            in {"auto", "name"}
        }
        unresolved = []
        for name in cls._extract_owner_candidates_from_commands(commands):
            name_key = cls._normalize_name_key(name)
            if name_key in resolved_map or name_key in in_flight:
                continue
            unresolved.append(name)
        return [
            {"type": "search_owners", "args": {"text": name, "mode": "name"}}
            for name in unresolved
        ]

    @classmethod
    def _rewrite_search_video_commands_with_owner_results(
        cls,
        commands: list[dict],
        last_tool_results: list[dict] | None,
    ) -> list[dict]:
        resolved_map = cls._extract_resolved_owner_map(last_tool_results)
        if not resolved_map:
            return commands

        rewritten = []
        for command in commands or []:
            if command.get("type") != "search_videos":
                rewritten.append(command)
                continue
            args = dict(command.get("args") or {})
            queries = args.get("queries")
            if isinstance(queries, str):
                queries = [queries]
            if not isinstance(queries, list) or not queries:
                rewritten.append(command)
                continue

            rewritten_queries: list[str] = []
            for query in queries:
                query_text = str(query or "")
                updated_query = query_text
                for name in cls._extract_user_filters_from_query(query_text):
                    resolved_owner = resolved_map.get(cls._normalize_name_key(name))
                    if not resolved_owner or not resolved_owner.get("mid"):
                        continue
                    updated_query = updated_query.replace(
                        f":user={name}",
                        f":uid={int(resolved_owner['mid'])}",
                    )
                if updated_query not in rewritten_queries:
                    rewritten_queries.append(updated_query)

            rewritten.append(
                {
                    "type": "search_videos",
                    "args": {
                        **args,
                        "queries": rewritten_queries,
                    },
                }
            )
        return rewritten

    @classmethod
    def _resolve_owner_result_scope(
        cls,
        owner_result_context: dict[str, dict],
        last_tool_results: list[dict] | None,
    ) -> list[dict] | None:
        if last_tool_results and all(
            result_item.get("type") == "search_owners"
            for result_item in last_tool_results
        ):
            return cls._filter_superseded_owner_results(
                list(owner_result_context.values())
            )
        return last_tool_results

    @classmethod
    def _merge_owner_resolution_commands(
        cls,
        commands: list[dict],
        owner_result_scope: list[dict] | None,
    ) -> list[dict]:
        owner_resolution_commands = cls._build_owner_resolution_commands(
            commands,
            owner_result_scope,
        )
        if not owner_resolution_commands:
            return commands
        passthrough_commands = [
            command for command in commands if command.get("type") != "search_videos"
        ]
        return passthrough_commands + owner_resolution_commands

    @classmethod
    def _plan_tool_commands(
        cls,
        commands: list[dict],
        messages: list[dict],
        last_tool_results: list[dict] | None,
        owner_result_scope: list[dict] | None,
    ) -> list[dict]:
        planned = cls._normalize_search_video_commands(commands)
        planned = cls._rewrite_search_video_commands_with_owner_results(
            planned,
            owner_result_scope,
        )
        planned = apply_tool_planning_plugins(
            cls,
            ToolPlanningContext(
                commands=planned,
                messages=messages,
                last_tool_results=last_tool_results,
                owner_result_scope=owner_result_scope,
            ),
            DEFAULT_TOOL_PLANNING_PLUGINS,
        )
        planned = cls._normalize_search_video_commands(planned)
        return cls._merge_owner_resolution_commands(planned, owner_result_scope)

    @classmethod
    def _extract_recent_assistant_commands(cls, messages: list[dict]) -> list[dict]:
        for message in reversed(messages or []):
            if message.get("role") != "assistant":
                continue
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            commands = cls._parse_tool_commands(content)
            if commands:
                return commands
        return []

    @staticmethod
    def _commands_target_videos(commands: list[dict] | None) -> bool:
        return any(command.get("type") == "search_videos" for command in commands or [])

    @classmethod
    def _wants_video_results(
        cls,
        messages: list[dict],
        commands: list[dict] | None = None,
    ) -> bool:
        if cls._commands_target_videos(commands):
            return True
        recent_commands = cls._extract_recent_assistant_commands(messages)
        if cls._commands_target_videos(recent_commands):
            return True
        if recent_commands and all(
            command.get("type") == "related_tokens_by_tokens"
            for command in recent_commands
        ):
            return True
        latest_user_text = cls._get_latest_user_text(messages)
        return bool(
            latest_user_text and _VIDEO_SEARCH_INTENT_RE.search(latest_user_text)
        )

    @classmethod
    def _wants_recent_video_results(
        cls,
        messages: list[dict],
        commands: list[dict] | None = None,
    ) -> bool:
        def has_date_filter(tool_commands: list[dict] | None) -> bool:
            for command in tool_commands or []:
                if command.get("type") != "search_videos":
                    continue
                queries = (command.get("args") or {}).get("queries")
                if isinstance(queries, str):
                    queries = [queries]
                for query in queries or []:
                    text = str(query)
                    if ":date<=" in text or ":date>=" in text:
                        return True
            return False

        if has_date_filter(commands):
            return True
        if has_date_filter(cls._extract_recent_assistant_commands(messages)):
            return True
        latest_user_text = cls._get_latest_user_text(messages)
        return bool(
            latest_user_text and _RECENT_VIDEO_INTENT_RE.search(latest_user_text)
        )

    @classmethod
    def _continue_intermediate_plan(
        cls,
        commands: list[dict],
        messages: list[dict],
        last_tool_results: list[dict] | None,
    ) -> list[dict]:
        if commands or not last_tool_results:
            return commands

        intermediate_tool_types = {
            result_item.get("type")
            for result_item in last_tool_results
            if result_item.get("type")
        }
        if not intermediate_tool_types.intersection(
            {"search_owners", "related_tokens_by_tokens"}
        ):
            return commands

        recovered = cls._extract_recent_assistant_commands(messages)
        if not recovered:
            return commands

        if any(
            result_item.get("type") == "search_owners"
            for result_item in last_tool_results or []
        ):
            recovered = [
                command
                for command in recovered
                if command.get("type") != "related_tokens_by_tokens"
            ]
            if not recovered:
                return commands

        recovered = cls._normalize_search_video_commands(recovered)
        recovered = cls._rewrite_search_video_commands_with_owner_results(
            recovered,
            last_tool_results,
        )
        recovered = cls._normalize_token_assisted_search_commands(
            recovered,
            messages,
            last_tool_results,
        )
        if not any(command.get("type") == "search_videos" for command in recovered):
            return commands
        return recovered

    @classmethod
    def _extract_token_rewrite_from_results(
        cls, results: list[dict] | None
    ) -> dict | None:
        for result_item in results or []:
            if result_item.get("type") != "related_tokens_by_tokens":
                continue
            result = result_item.get("result") or {}
            source_text = str(
                result.get("text") or result_item.get("args", {}).get("text") or ""
            ).strip()
            if not source_text:
                continue
            candidates: list[str] = []
            for option in result.get("options") or []:
                candidate = str(option.get("text") or "").strip()
                if (
                    candidate
                    and candidate != source_text
                    and candidate not in candidates
                ):
                    candidates.append(candidate)
            if candidates:
                return {
                    "source_text": source_text,
                    "candidates": candidates,
                }
        return None

    @staticmethod
    def _canonicalize_token_option_key(text: str) -> str:
        return _TOKEN_OPTION_CANONICAL_RE.sub("", (text or "").strip()).lower()

    @classmethod
    def _select_distinct_token_candidates(
        cls,
        source_text: str,
        candidates: list[str] | None,
        limit: int = 3,
    ) -> list[str]:
        source_key = cls._canonicalize_token_option_key(source_text)
        distinct: list[str] = []
        seen_keys: set[str] = set()
        for candidate in candidates or []:
            key = cls._canonicalize_token_option_key(candidate)
            if not key or key == source_key or key in seen_keys:
                continue
            seen_keys.add(key)
            distinct.append(candidate)
            if len(distinct) >= limit:
                break
        return distinct

    @classmethod
    def _build_token_assisted_video_queries(
        cls,
        messages: list[dict],
        token_rewrite: dict | None,
        *,
        commands: list[dict] | None = None,
    ) -> list[str]:
        if not token_rewrite:
            return []
        latest_user_text = cls._get_latest_user_text(messages)
        if not latest_user_text or not cls._wants_video_results(messages, commands):
            return []

        source_text = str(token_rewrite.get("source_text") or "").strip()
        candidates = cls._select_distinct_token_candidates(
            source_text,
            token_rewrite.get("candidates") or [],
            limit=4,
        )
        if not source_text or not candidates:
            return []

        residual_text = latest_user_text.replace(source_text, " ")
        residual_text = cls._normalize_entity_focused_query_text(residual_text)
        source_query = cls._normalize_entity_focused_query_text(source_text)
        residual_is_weak = (
            not residual_text
            or residual_text == source_query
            or len(cls._normalize_name_key(residual_text)) <= 2
        )

        query_candidates = candidates[:3] if residual_is_weak else candidates[:1]
        queries: list[str] = []
        for candidate in query_candidates:
            candidate_text = (
                cls._normalize_entity_focused_query_text(candidate) or candidate
            )
            query = (
                candidate_text
                if residual_is_weak
                else f"{candidate_text} {residual_text}".strip()
            )
            query = re.sub(r"\s+", " ", query).strip()
            if not query:
                continue
            if "q=" not in query and ":user=" not in query and ":uid=" not in query:
                query = f"{query} q=vwr"
            if query not in queries:
                queries.append(query)
        return queries

    @classmethod
    def _normalize_token_assisted_search_commands(
        cls,
        commands: list[dict],
        messages: list[dict],
        last_tool_results: list[dict] | None,
    ) -> list[dict]:
        token_rewrite = cls._extract_token_rewrite_from_results(last_tool_results)
        if not token_rewrite:
            return commands

        source_text = str(token_rewrite.get("source_text") or "").strip()
        replacement_queries = cls._build_token_assisted_video_queries(
            messages,
            token_rewrite,
            commands=commands,
        )
        if not replacement_queries:
            return commands

        normalized = []
        for command in commands or []:
            if command.get("type") != "search_videos":
                normalized.append(command)
                continue
            args = dict(command.get("args") or {})
            queries = args.get("queries")
            if isinstance(queries, str):
                queries = [queries]
            if not isinstance(queries, list) or not queries:
                normalized.append(command)
                continue
            if not any(source_text in str(query) for query in queries):
                normalized.append(command)
                continue
            rewritten_queries: list[str] = []
            for query in queries:
                if source_text in str(query):
                    for replacement_query in replacement_queries:
                        if replacement_query not in rewritten_queries:
                            rewritten_queries.append(replacement_query)
                    continue
                if query not in rewritten_queries:
                    rewritten_queries.append(query)
            normalized.append(
                {
                    "type": "search_videos",
                    "args": {
                        **args,
                        "queries": rewritten_queries,
                    },
                }
            )
        return normalized

    @classmethod
    def _fallback_token_assisted_search_commands(
        cls,
        commands: list[dict],
        messages: list[dict],
        last_tool_results: list[dict] | None,
    ) -> list[dict]:
        if commands or not last_tool_results:
            return commands
        token_rewrite = cls._extract_token_rewrite_from_results(last_tool_results)
        queries = cls._build_token_assisted_video_queries(messages, token_rewrite)
        if not queries:
            return commands
        return [{"type": "search_videos", "args": {"queries": queries}}]
