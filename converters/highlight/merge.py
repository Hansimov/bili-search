import re


class HighlightMerger:
    def merge(self, text: str, htexts: list[str], tag: str = "em"):
        """Merge all highlighted text segments into one."""
        if not text or not htexts:
            return None

        pattern = f"<{tag}>(.*?)</{tag}>"
        highlighted_segements = set()
        for htext in htexts:
            segments = re.findall(pattern, htext)
            highlighted_segements.update(segments)

        res_text = text
        for seg in highlighted_segements:
            res_text = res_text.replace(seg, f"<{tag}>{seg}</{tag}>")
        return res_text
