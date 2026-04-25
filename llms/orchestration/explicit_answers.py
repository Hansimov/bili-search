from __future__ import annotations

from llms.contracts import IntentProfile
from llms.orchestration.policies import has_explicit_video_anchor
from llms.orchestration.policies import is_recent_timeline_request
from llms.orchestration.video_queries import VideoQueryNormalizer
from llms.tools.names import canonical_tool_name


class ExplicitDeterministicAnswerMixin:
    def _build_deterministic_final_answer(
        self,
        intent: IntentProfile,
        messages: list[dict],
    ) -> str | None:
        return (
            self._build_explicit_dsl_video_search_answer(intent)
            or self._build_explicit_video_lookup_answer(intent)
            or self._build_owner_recent_timeline_answer(
                intent,
                messages,
            )
        )

    def _build_explicit_dsl_video_search_answer(
        self,
        intent: IntentProfile,
    ) -> str | None:
        explicit_query = VideoQueryNormalizer.extract_explicit_dsl_query(
            intent.raw_query
        )
        if not explicit_query:
            return None

        best_result: dict | None = None
        for result_id in self.result_store.order:
            record = self.result_store.get(result_id)
            if (
                record is None
                or canonical_tool_name(record.request.name) != "search_videos"
            ):
                continue
            args = record.request.arguments or {}
            queries = args.get("queries")
            if isinstance(queries, str):
                query_values = [queries]
            elif isinstance(queries, list):
                query_values = [str(item or "") for item in queries]
            else:
                query_values = []
            if explicit_query not in query_values:
                continue
            best_result = record.result or {}
            break

        if not best_result:
            return None
        hits = list(best_result.get("hits") or [])
        total_hits = int(best_result.get("total_hits") or len(hits))
        if not hits:
            return f"按 `{explicit_query}` 搜索后，当前没有找到可展示的视频结果。"

        lines = [f"按 `{explicit_query}` 找到这些相关视频："]
        for index, hit in enumerate(hits[:5], start=1):
            title = str(hit.get("title") or "").strip() or "未命名视频"
            bvid = str(hit.get("bvid") or "").strip()
            owner = hit.get("owner") or {}
            owner_name = owner.get("name", "") if isinstance(owner, dict) else ""
            suffix_parts = []
            if owner_name:
                suffix_parts.append(f"UP：{owner_name}")
            if bvid:
                suffix_parts.append(f"https://www.bilibili.com/video/{bvid}")
            suffix = f"（{'，'.join(suffix_parts)}）" if suffix_parts else ""
            lines.append(f"{index}. 《{title}》{suffix}")

        if total_hits > len(hits):
            lines.append(f"当前展示前 {len(hits)} 条，可检索总命中约 {total_hits} 条。")
        return "\n".join(lines).strip()

    def _build_explicit_video_lookup_answer(
        self,
        intent: IntentProfile,
    ) -> str | None:
        if intent.final_target != "videos" or not has_explicit_video_anchor(intent):
            return None
        if not (intent.needs_owner_resolution or is_recent_timeline_request(intent)):
            return None

        primary_hit: dict | None = None
        recent_hits: list[dict] = []

        for result_id in self.result_store.order:
            record = self.result_store.get(result_id)
            if (
                record is None
                or canonical_tool_name(record.request.name) != "search_videos"
            ):
                continue

            result = record.result or {}
            lookup_by = str(result.get("lookup_by") or "").lower()
            hits = result.get("hits") or []
            if not primary_hit and lookup_by in {"bvid", "bvids"} and hits:
                primary_hit = hits[0]
            if lookup_by in {"mid", "mids"} and not recent_hits:
                recent_hits = list(hits)

        if not primary_hit:
            return None

        owner = primary_hit.get("owner") or {}
        bvid = str(primary_hit.get("bvid") or "").strip()
        title = str(primary_hit.get("title") or "").strip()
        owner_name = str(owner.get("name") or "").strip()
        owner_mid = str(owner.get("mid") or "").strip()

        lines: list[str] = []
        title_text = f"《{title}》" if title else "该视频"
        if bvid and title:
            lines.append(f"{bvid} 这期视频的标题是 {title_text}。")
        elif title:
            lines.append(f"这期视频的标题是 {title_text}。")

        if owner_name and owner_mid:
            lines.append(f"作者是 {owner_name}，UID 为 {owner_mid}。")
        elif owner_name:
            lines.append(f"作者是 {owner_name}。")
        elif owner_mid:
            lines.append(f"作者 UID 为 {owner_mid}。")

        if is_recent_timeline_request(intent):
            if recent_hits:
                lines.append("该作者近 30 天发布的视频包括：")
                for index, hit in enumerate(recent_hits[:5], start=1):
                    hit_title = str(hit.get("title") or "").strip()
                    hit_bvid = str(hit.get("bvid") or "").strip()
                    if hit_title and hit_bvid:
                        lines.append(f"{index}. 《{hit_title}》({hit_bvid})")
                    elif hit_title:
                        lines.append(f"{index}. 《{hit_title}》")
                    elif hit_bvid:
                        lines.append(f"{index}. {hit_bvid}")
            else:
                lines.append("当前 30 天时间窗内未检索到该作者的其他公开视频。")

        return "\n".join(line for line in lines if line).strip() or None
