import argparse
import concurrent.futures
import sys

from copy import deepcopy
from elasticsearch.helpers import bulk
from tclogger import logger
from tqdm import tqdm

from elastics.client import ElasticSearchClient
from networks.mongo import MongoOperator

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
        self.mongo = MongoOperator()

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

    def mongo_doc_to_elastic_doc(self, doc):
        mongo_doc = deepcopy(doc)
        doc_id = mongo_doc.pop("_id")
        elastic_doc = {
            "_index": self.index_name,
            "_id": doc_id,
            "_source": mongo_doc,
        }
        return elastic_doc

    def index_docs_to_elastic(self, docs: list, max_workers: int = 4):
        doc_chunks = [docs[i::max_workers] for i in range(max_workers)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(bulk, self.es.client, docs_chunk)
                for docs_chunk in doc_chunks
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    def get_not_existent_doc_ids(self, doc_ids: list):
        if not doc_ids:
            return []
        response = self.es.client.mget(
            index=self.index_name, ids=doc_ids, _source=False
        )
        not_existent_ids = [doc["_id"] for doc in response["docs"] if not doc["found"]]
        return not_existent_ids

    def index_docs_from_mongo_to_elastic(
        self,
        mongo_collection: str = "videos",
        max_count: int = None,
        batch_size: int = 100,
        max_workers: int = 4,
        overwrite: bool = True,
    ):
        logger.note(f"> Indexing docs:", end=" ")
        logger.file(f"[{mongo_collection}] (mongo) -> [{self.index_name}] (elastic)")

        total_count = self.mongo.db[mongo_collection].estimated_document_count()
        cursor = self.mongo.get_cursor(collection=mongo_collection, sort_index="aid")
        progress_bar = tqdm(cursor, total=max_count or total_count)

        docs_batch = []
        for idx, doc in enumerate(progress_bar):
            if max_count and idx + 1 > max_count:
                break
            pubdate_str = doc["pubdate_str"]
            progress_desc = f"{pubdate_str}, aid: {doc['aid']}"
            progress_bar.set_description(progress_desc)

            elastic_doc = self.mongo_doc_to_elastic_doc(doc)
            docs_batch.append(elastic_doc)

            if len(docs_batch) >= batch_size:
                if not overwrite:
                    doc_ids = [doc["_id"] for doc in docs_batch]
                    not_existent_ids = set(self.get_not_existent_doc_ids(doc_ids))
                    if not_existent_ids:
                        docs_batch = [
                            doc for doc in docs_batch if doc["_id"] in not_existent_ids
                        ]
                self.index_docs_to_elastic(docs_batch, max_workers=max_workers)
                docs_batch = []

        if docs_batch:
            self.index_docs_to_elastic(docs_batch, max_workers=max_workers)


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

    indexer.index_docs_from_mongo_to_elastic(
        "videos", batch_size=1000 * 16, max_workers=16
    )

    # python -m elastics.video_indexer
    # python -m elastics.video_indexer -i bili_videos_dev
    # python -m elastics.video_indexer -i bili_videos_dev -r
