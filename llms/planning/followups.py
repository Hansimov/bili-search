from __future__ import annotations

from llms.intent import build_intent_profile
from llms.intent.focus import compact_focus_key
from llms.messages import normalize_bvid_key
from llms.tools.names import canonical_tool_name


def _tool_type(item: dict) -> str:
    return canonical_tool_name(str(item.get("type") or ""))


class ToolFollowupPlanningMixin:
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
            _tool_type(command) == "expand_query" for command in recent_commands
        ):
            return True
        intent = build_intent_profile(messages)
        return intent.final_target in {"videos", "mixed"}

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
        latest_user_text = "".join(cls._get_latest_user_text(messages).split())
        if any(
            token in latest_user_text
            for token in (
                "最近",
                "近期",
                "近况",
                "最近还发",
                "最近还有哪些视频",
                "最近发了哪些视频",
            )
        ):
            return True
        intent = build_intent_profile(messages)
        return intent.task_mode == "repeat" or "recent_only" in intent.top_labels(
            "constraints",
            limit=8,
        )

    @classmethod
    def _extract_explicit_video_lookup_owner_context(
        cls,
        results: list[dict] | None,
    ) -> tuple[list[int], list[str], list[str]]:
        mids: list[int] = []
        owner_names: list[str] = []
        anchor_bvids: list[str] = []
        anchor_bvid_keys: set[str] = set()

        for result_item in results or []:
            if _tool_type(result_item) != "search_videos":
                continue

            args = result_item.get("args") or {}
            result = result_item.get("result") or {}
            lookup_by = str(result.get("lookup_by") or "").lower()
            has_bvid_seed = bool(
                args.get("bv") or args.get("bvid") or args.get("bvids")
            )
            if lookup_by not in {"bvids", "bvid"} and not has_bvid_seed:
                continue

            for value in args.get("bvids") or []:
                bvid = str(value or "").strip()
                bvid_key = normalize_bvid_key(bvid)
                if bvid_key and bvid_key not in anchor_bvid_keys:
                    anchor_bvid_keys.add(bvid_key)
                    anchor_bvids.append(bvid)
            for key in ("bv", "bvid"):
                bvid = str(args.get(key, "") or "").strip()
                bvid_key = normalize_bvid_key(bvid)
                if bvid_key and bvid_key not in anchor_bvid_keys:
                    anchor_bvid_keys.add(bvid_key)
                    anchor_bvids.append(bvid)
            for value in result.get("bvids") or []:
                bvid = str(value or "").strip()
                bvid_key = normalize_bvid_key(bvid)
                if bvid_key and bvid_key not in anchor_bvid_keys:
                    anchor_bvid_keys.add(bvid_key)
                    anchor_bvids.append(bvid)

            for hit in result.get("hits") or []:
                owner = hit.get("owner") or {}
                owner_name = str(owner.get("name") or "").strip()
                if owner_name and owner_name not in owner_names:
                    owner_names.append(owner_name)
                owner_mid = owner.get("mid")
                try:
                    normalized_mid = int(str(owner_mid).strip())
                except (TypeError, ValueError):
                    normalized_mid = None
                if normalized_mid and normalized_mid not in mids:
                    mids.append(normalized_mid)

        return mids, owner_names, anchor_bvids

    @classmethod
    def _build_explicit_video_lookup_followup_commands(
        cls,
        commands: list[dict],
        messages: list[dict],
        last_tool_results: list[dict] | None,
        intent=None,
    ) -> list[dict]:
        if commands or not last_tool_results:
            return commands
        if not cls._wants_recent_video_results(messages):
            return commands

        mids, owner_names, anchor_bvids = (
            cls._extract_explicit_video_lookup_owner_context(last_tool_results)
        )
        if not mids and not owner_names:
            return commands

        window = cls._extract_recent_window(cls._get_latest_user_text(messages))
        if mids:
            args: dict = {
                "mode": "lookup",
                "date_window": window,
                "limit": 10,
            }
            if anchor_bvids:
                args["exclude_bvids"] = anchor_bvids[:10]
            if len(mids) == 1:
                args["mid"] = str(mids[0])
            else:
                args["mids"] = [str(mid) for mid in mids[:5]]
            return [{"type": "search_videos", "args": args}]

        queries: list[str] = []
        for owner_name in owner_names[:3]:
            query = f":user={owner_name} :date<={window}"
            if query not in queries:
                queries.append(query)
        if not queries:
            return commands
        return [{"type": "search_videos", "args": {"queries": queries}}]

    @classmethod
    def _continue_intermediate_plan(
        cls,
        commands: list[dict],
        messages: list[dict],
        last_tool_results: list[dict] | None,
        intent=None,
    ) -> list[dict]:
        if commands or not last_tool_results:
            return commands

        intermediate_tool_types = {
            result_item.get("type")
            for result_item in last_tool_results
            if result_item.get("type")
        }
        if not intermediate_tool_types.intersection({"search_owners", "expand_query"}):
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
                if _tool_type(command) != "expand_query"
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
            intent,
        )
        if not any(command.get("type") == "search_videos" for command in recovered):
            return commands
        return recovered

    @classmethod
    def _extract_token_rewrite_from_results(
        cls, results: list[dict] | None
    ) -> dict | None:
        for result_item in results or []:
            if _tool_type(result_item) != "expand_query":
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
        return compact_focus_key(text)

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
            query = " ".join(query.split()).strip()
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
        intent=None,
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
        intent=None,
    ) -> list[dict]:
        if commands or not last_tool_results:
            return commands
        token_rewrite = cls._extract_token_rewrite_from_results(last_tool_results)
        queries = cls._build_token_assisted_video_queries(messages, token_rewrite)
        if not queries:
            return commands
        return [{"type": "search_videos", "args": {"queries": queries}}]
