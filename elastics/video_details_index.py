import json

from tclogger import logger
from tqdm import tqdm

from configs.envs import BILI_DATA_ROOT
from elastics.client import ElasticSearchClient

VIDEO_DETAILS_INDEX_SETTINGS = {
    "analysis": {
        "analyzer": {
            "word_analyzer": {"type": "ik_max_word"},
            "pinyin_analyzer": {"type": "custom", "tokenizer": "pinyin_tokenizer"},
        },
        "tokenizer": {
            "pinyin_tokenizer": {
                "type": "pinyin",
                "keep_first_letter": True,
                "keep_separate_first_letter": False,
                "keep_full_pinyin": True,
                "keep_joined_full_pinyin": True,
                "keep_original": True,
                "limit_first_letter_length": 16,
                "lowercase": True,
                "remove_duplicated_term": True,
                "ignore_pinyin_offset": True,
            }
        },
    }
}

VIDEO_DETAILS_INDEX_MAPPINGS = {
    "dynamic_templates": [
        {
            "pinyin_template": {
                "match": "^(title|owner.name|desc|tname|dynamic|pages.part)$",
                "match_pattern": "regex",
                "mapping": {"type": "text", "analyzer": "pinyin_analyzer"},
            }
        },
        {
            "datetime_template": {
                "match": "^(pubdate|ctime)$",
                "match_pattern": "regex",
                "mapping": {"type": "date", "format": "epoch_second"},
            }
        },
        {
            "rights_template": {
                "path_match": "rights.*",
                "mapping": {"type": "byte"},
            }
        },
        {
            "string_template": {
                "match_mapping_type": "string",
                "mapping": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword"}},
                },
            }
        },
    ]
}


class VideoDetailsIndexer:
    def __init__(self, index_name: str = "bili_video_details"):
        self.index_name = index_name
        self.es = ElasticSearchClient()
        self.es.connect()

    def create_index(self, delete_old: bool = True):
        logger.note(f"> Creating index:", end=" ")
        logger.mesg(f"[{self.index_name}]")
        if delete_old:
            self.es.client.indices.delete(
                index=self.index_name, ignore_unavailable=True
            )
        self.es.client.indices.create(
            index=self.index_name,
            settings=VIDEO_DETAILS_INDEX_SETTINGS,
            mappings=VIDEO_DETAILS_INDEX_MAPPINGS,
        )

    def udpate_docs(self, mid: int = None):
        video_details_dir = BILI_DATA_ROOT / f"{mid}" / "video_details"
        logger.note("> Updating video details docs ...")
        for video_details_path in tqdm(sorted(video_details_dir.glob("*.json"))):
            with open(video_details_path, "r") as rf:
                video_details = json.load(rf)
            bvid = video_details.get("bvid", None)
            if bvid:
                self.es.client.index(
                    index=self.index_name,
                    id=bvid,
                    body=video_details,
                )
            else:
                logger.warn(f"× No details for file:")
                logger.file(f"  - [{video_details_path}]")


if __name__ == "__main__":
    indexer = VideoDetailsIndexer()
    indexer.create_index()
    mid = 946974
    indexer.udpate_docs(mid)

    # python -m elastics.video_details_index
