"""Owner search result parser.

Converts raw ES response into clean owner hit dicts for API or internal use.
"""

from elastics.owners.constants import SOURCE_FIELDS, SOURCE_FIELDS_COMPACT


class OwnerHitsParser:
    """Parses raw ES search responses for the owners index."""

    def parse_hit(self, hit: dict, compact: bool = False) -> dict:
        """Parse a single ES hit into a clean owner dict.

        Args:
            hit: Raw ES hit dict with _source, _score, etc.
            compact: If True, return only compact fields.

        Returns:
            Clean owner dict with selected fields + _score.
        """
        source = hit.get("_source", {})
        score = hit.get("_score")

        allowed = SOURCE_FIELDS_COMPACT if compact else SOURCE_FIELDS
        result = {k: source.get(k) for k in allowed if source.get(k) is not None}

        if score is not None:
            result["_score"] = round(score, 4)

        return result

    def parse_response(
        self,
        response: dict,
        compact: bool = False,
    ) -> dict:
        """Parse a full ES search response.

        Args:
            response: Raw ES search response body.
            compact: If True, return compact owner dicts.

        Returns:
            Dict with:
                hits: List of parsed owner dicts.
                total: Total hit count.
                max_score: Maximum score among hits.
        """
        hits_data = response.get("hits", {})
        raw_hits = hits_data.get("hits", [])
        total_obj = hits_data.get("total", {})

        if isinstance(total_obj, dict):
            total = total_obj.get("value", 0)
        else:
            total = total_obj or 0

        max_score = hits_data.get("max_score")

        parsed_hits = [self.parse_hit(h, compact=compact) for h in raw_hits]

        return {
            "hits": parsed_hits,
            "total": total,
            "max_score": round(max_score, 4) if max_score is not None else None,
        }

    def format_for_api(self, parsed: dict) -> dict:
        """Format parsed results for API response (frontend consumption).

        Adds human-readable fields and summaries.
        """
        hits = parsed.get("hits", [])
        for hit in hits:
            # Add human-readable view count
            total_view = hit.get("total_view", 0)
            if total_view >= 1e8:
                hit["total_view_str"] = f"{total_view / 1e8:.1f}亿"
            elif total_view >= 1e4:
                hit["total_view_str"] = f"{total_view / 1e4:.1f}万"
            else:
                hit["total_view_str"] = str(total_view)

            # Add human-readable days_since_last
            days = hit.get("days_since_last", 0)
            if days is not None:
                if days <= 0:
                    hit["last_active_str"] = "今天"
                elif days <= 1:
                    hit["last_active_str"] = "昨天"
                elif days <= 7:
                    hit["last_active_str"] = f"{days}天前"
                elif days <= 30:
                    hit["last_active_str"] = f"{days // 7}周前"
                elif days <= 365:
                    hit["last_active_str"] = f"{days // 30}个月前"
                else:
                    hit["last_active_str"] = f"{days // 365}年前"

        return parsed
