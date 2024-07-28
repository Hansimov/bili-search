import argparse
import json
import sys

from tclogger import logger, ts_to_str
from tqdm import tqdm

from configs.envs import BILI_DATA_ROOT
from elastics.client import ElasticSearchClient

# https://github.com/infinilabs/analysis-pinyin
VIDEO_DETAILS_INDEX_SETTINGS = {
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

VIDEO_DETAILS_INDEX_MAPPINGS = {
    "dynamic_templates": [
        {
            "suggest_template": {
                "path_match": "^(title|owner.name|pages.part|honor_reply.honor.desc)$",
                "match_pattern": "regex",
                "mapping": {
                    "type": "text",
                    "analyzer": "chinese_analyzer",
                    "search_analyzer": "chinese_analyzer",
                    "fields": {
                        "pinyin": {
                            "type": "text",
                            "analyzer": "pinyin_analyzer",
                        },
                        "pinyin_suggest": {
                            "type": "completion",
                            "analyzer": "pinyin_analyzer",
                        },
                        "text_suggest": {
                            "type": "completion",
                            "analyzer": "chinese_analyzer",
                            "search_analyzer": "chinese_analyzer",
                        },
                    },
                },
            }
        },
        {
            "pinyin_template": {
                "path_match": "^(title|owner.name|desc|tname|dynamic|pages.part|honor_reply.honor.desc)$",
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
                "path_match": "^(pubdate|ctime)$",
                "match_pattern": "regex",
                "mapping": {
                    "type": "date",
                    "format": "epoch_second",
                },
            }
        },
        {
            "datetime_str_template": {
                "path_match": "^(pubdate_str|ctime_str)$",
                "match_pattern": "regex",
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
        {
            "rights_template": {
                "path_match": "rights.*",
                "mapping": {
                    "type": "byte",
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
                        "keyword": {
                            "type": "keyword",
                        },
                    },
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

    def create_index(self, delete_old: bool = False):
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
                for key in ["pubdate", "ctime"]:
                    if video_details.get(key, None):
                        video_details[f"{key}_str"] = ts_to_str(video_details.get(key))
                self.es.client.index(
                    index=self.index_name,
                    id=bvid,
                    body=video_details,
                )
            else:
                logger.warn(f"× No details for file:")
                logger.file(f"  - [{video_details_path}]")

    def update_datetime_str_field(self, keys: list[str] = ["pubdate", "ctime"]):
        for key in keys:
            logger.note(
                f"> Add new field in existing docs: datetime timestamp '{key}' to string '{key}_str'"
            )
            script_source = f"""
            if (ctx._source.containsKey('{key}')) {{
                Instant instant = Instant.ofEpochSecond(ctx._source.{key});
                ZonedDateTime zdt = ZonedDateTime.ofInstant(instant, ZoneId.of('UTC+8'));
                DateTimeFormatter formatter = DateTimeFormatter.ofPattern('yyyy-MM-dd HH:mm:ss');
                ctx._source.{key}_str = formatter.format(zdt);
            }}
            """
            query = {"script": {"source": script_source, "lang": "painless"}}
            logger.mesg(script_source)
            self.es.client.update_by_query(
                index=self.index_name,
                body=query,
                wait_for_completion=True,
                conflicts="proceed",
            )

    def delete_field_from_doc(self, field: str):
        logger.note(f"> Delete field: {field}")
        script_source = f"ctx._source.remove('{field}')"
        query = {
            "script": {
                "source": script_source,
                "lang": "painless",
            }
        }
        logger.mesg(script_source)
        self.es.client.update_by_query(
            index=self.index_name,
            body=query,
            wait_for_completion=True,
            conflicts="proceed",
        )

    def put_mapping(self, field: str, mapping: dict):
        logger.note(f"> Add new mapping for field: {field}")
        query = {"properties": {field: mapping}}
        self.es.client.indices.put_mapping(index=self.index_name, body=query)


class ArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(ArgParser, self).__init__(*args, **kwargs)
        self.add_argument(
            "-m",
            "--mid",
            type=int,
            help="User mid",
        )
        self.add_argument(
            "-i",
            "--index",
            type=str,
            default="bili_video_details",
            help="Index name",
        )
        self.add_argument(
            "-r",
            "--recreate",
            action="store_true",
            help="Delete and recreate existed index of elasticsearch",
        )

        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])


if __name__ == "__main__":
    args = ArgParser().args
    indexer = VideoDetailsIndexer(index_name=args.index)

    if args.recreate:
        indexer.create_index(args.recreate)

    # mid = args.mid or 946974
    # indexer.udpate_docs(mid)

    mids = [
        946974,
        12566101,
        14871346,
        18937923,
        20259914,
        46377861,
        1018175068,
        1369507485,
    ]
    for mid in mids:
        indexer.udpate_docs(mid)

    # for field in ["pubdate_str", "ctime_str"]:
    #     indexer.delete_field_from_doc(field)

    # indexer.put_mapping(
    #     "pubdate_str",
    #     {
    #         "type": "text",
    #         "analyzer": "chinese_analyzer",
    #         "fields": {
    #             "string": {
    #                 "type": "text",
    #                 "analyzer": "whitespace",
    #             },
    #         },
    #     },
    # )
    # indexer.update_datetime_str_field(["pubdate", "ctime"])

    # Just run the main funciton:
    # python -m elastics.video_details_indexer

    # Index video details in production index (with recreate):
    # python -m elastics.video_details_indexer -i bili_video_details_dev -r

    # Index video details in dev index for mid:
    # python -m elastics.video_details_indexer -i bili_video_details_dev -r -m 12566101
