import re


class HighlightMerger:
    def merge(self, text: str, htexts: list[str], tag: str = "em"):
        """Merge all highlighted text segments into one."""
        if not text or not htexts:
            return None

        # get all highlighted segments
        hpattern = f"<{tag}>(.*?)</{tag}>"
        hsegs = set()
        for htext in htexts:
            segments = re.findall(hpattern, htext)
            hsegs.update(segments)

        # get indexes of highlighted segments in text
        hindexes = []
        for hseg in hsegs:
            for match in re.finditer(hseg, text):
                hindexes.append((match.start(), match.end()))

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
        res_text = ""
        last_index = 0
        for start, end in merged_indexes:
            res_text += text[last_index:start]
            res_text += f"<{tag}>{text[start:end]}</{tag}>"
            last_index = end
        res_text += text[last_index:]

        return res_text


if __name__ == "__main__":
    from tclogger import logger

    text = "影、视业的发展"
    htexts = [
        "<hit>影、视</hit>业的发展",
        "影<hit>、视业</hit>的发展",
        "<hit>影、视业</hit>的发展",
        "<hit>影、</hit>视业的发展",
    ]

    merger = HighlightMerger()
    merged_text = merger.merge(text, htexts, tag="hit")
    logger.success(merged_text)

    # python -m converters.highlight.merge
