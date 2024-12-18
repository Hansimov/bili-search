import re

from typing import Union


class HighlightMerger:
    def extract_and_merge(
        self, text: str, htexts: list[str], tag: str = "hit"
    ) -> dict[str, Union[str, list[str]]]:
        """Extract all highlighted segments, and merge into one."""
        if not text or not htexts:
            return {"segged": [], "merged": ""}

        # get all highlighted segments
        hpattern = f"<{tag}>(.*?)</{tag}>"
        hsegs = set()
        for htext in htexts:
            segments = re.findall(hpattern, htext)
            hsegs.update(segments)

        # get indexes of highlighted segments in text
        hindexes = []
        for hseg in hsegs:
            try:
                for match in re.finditer(re.escape(hseg), text):
                    hindexes.append((match.start(), match.end()))
            except Exception as e:
                pass

        # combine overlapping indexes
        hindexes.sort()
        merged_indexes = []
        for start, end in hindexes:
            if merged_indexes and merged_indexes[-1][1] >= start:
                merged_indexes[-1] = (
                    merged_indexes[-1][0],
                    max(merged_indexes[-1][1], end),
                )
            else:
                merged_indexes.append((start, end))

        # insert tags by indexes
        merged_text = ""
        last_index = 0
        for start, end in merged_indexes:
            merged_text += text[last_index:start]
            merged_text += f"<{tag}>{text[start:end]}</{tag}>"
            last_index = end
        merged_text += text[last_index:]

        res = {"segged": list(hsegs), "merged": merged_text}
        return res


if __name__ == "__main__":
    from tclogger import logger, dict_to_str

    text = "影、视业的发展"
    htexts = [
        "<hit>影、视</hit>业的发展",
        "影<hit>、视业</hit>的发展",
        "<hit>影、视业</hit>的发展",
        "<hit>影、</hit>视业的发展",
    ]

    merger = HighlightMerger()
    merged_res = merger.extract_and_merge(text, htexts, tag="hit")
    logger.success(dict_to_str(merged_res))

    # python -m converters.highlight.merge
