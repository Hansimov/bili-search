import argparse
import json
import sys

from datetime import datetime
from tclogger import logger
from tqdm import tqdm

from configs.envs import BILI_DATA_ROOT
from elastics.client import ElasticSearchClient

VIDEO_INDEX_SETTINGS = {
    "analysis": {
        "analyzer": {
            "chinese_analyzer": {
                "type": "custom",
                "tokenizer": "ik_max_word",
                "char_filter": "tscovert_char_filter",
            },
            "chinese_search_analyzer": {
                "type": "custom",
                "tokenizer": "ik_smart",
                "char_filter": "tscovert_char_filter",
            },
            "pinyin_analyzer": {
                "type": "custom",
                "tokenizer": "pinyin_tokenizer",
                "char_filter": "tscovert_char_filter",
            },
            "whitespace_analyzer": {
                "type": "custom",
                "tokenizer": "whitespace",
                "filter": ["lowercase"],
            },
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
        "char_filter": {
            "tscovert_char_filter": {
                "type": "stconvert",
                "convert_type": "t2s",
            }
        },
    }
}

VIDEO_INDEX_MAPPINGS = {
    "dynamic_templates": [
        {
            "pinyin_template": {
                "path_match": "^(title|owner.name|desc|tname)$",
                "match_pattern": "regex",
                "mapping": {
                    "type": "text",
                    "analyzer": "chinese_analyzer",
                    "search_analyzer": "chinese_analyzer",
                    "fields": {
                        "pinyin": {
                            "type": "text",
                            "analyzer": "pinyin_analyzer",
                        }
                    },
                },
            }
        },
        {
            "datetime_template": {
                "path_match": "^(pubdate|ctime|insert_at)$",
                "match_pattern": "regex",
                "mapping": {
                    "type": "date",
                    "format": "epoch_second",
                },
            }
        },
        {
            "datetime_str_template": {
                "path_match": "^(pubdate_str|ctime_str|insert_at_str)$",
                "match_pattern": "regex",
                "mapping": {
                    "type": "text",
                    "analyzer": "whitespace_analyzer",
                },
            }
        },
        {
            "string_template": {
                "match_mapping_type": "string",
                "mapping": {
                    "type": "text",
                    "analyzer": "chinese_analyzer",
                    "search_analyzer": "chinese_analyzer",
                    "fields": {
                        "string": {
                            "type": "text",
                            "analyzer": "whitespace_analyzer",
                        },
                    },
                },
            }
        },
    ]
}


class VideoIndexer:
    def __init__(self, index_name: str = "bili_videos"):
        self.index_name = index_name
        self.es = ElasticSearchClient()
        self.es.connect()

    def create_index(self, rewrite: bool = False):
        logger.note(f"> Creating index:", end=" ")
        logger.mesg(f"[{self.index_name}]")
        try:
            if rewrite:
                self.es.client.indices.delete(
                    index=self.index_name, ignore_unavailable=True
                )
            self.es.client.indices.create(
                index=self.index_name,
                settings=VIDEO_INDEX_SETTINGS,
                mappings=VIDEO_INDEX_MAPPINGS,
            )
        except Exception as e:
            logger.warn(f"× Error: {e}")


class ArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(ArgParser, self).__init__(*args, **kwargs)
        self.add_argument(
            "-i",
            "--index",
            type=str,
            default="bili_videos",
            help="Index name",
        )
        self.add_argument(
            "-r",
            "--rewrite",
            action="store_true",
            help="Delete and rewrite existed index of elasticsearch",
        )

        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])


if __name__ == "__main__":
    args = ArgParser().args
    indexer = VideoIndexer(index_name=args.index)

    if args.rewrite:
        indexer.create_index(args.rewrite)

    # python -m elastics.video_indexer
    # python -m elastics.video_indexer -i bili_videos_dev -r
