import bson.timestamp
import pymongo

from datetime import datetime
from pprint import pformat
import pymongo.errors
from tclogger import logger

from configs.envs import MONGO_ENVS


class MongoOperator:
    def __init__(self):
        self.host = MONGO_ENVS["host"]
        self.port = MONGO_ENVS["port"]
        self.dbname = MONGO_ENVS["dbname"]
        self.endpoint = f"mongodb://{self.host}:{self.port}/"
        self.connect()

    def connect(self):
        logger.note(f"> Connecting to: {self.endpoint} ...")
        self.client = pymongo.MongoClient(self.endpoint)
        self.db = self.client[self.dbname]
        logger.success(f"+ Connected to [{self.dbname}]")

    def get_cursor(
        self,
        collection: str,
        sort_index: str = None,
        sort_order: str = pymongo.ASCENDING,
        index_offset: int = None,
    ):
        if sort_index:
            if index_offset:
                filter = {sort_index: {"$gte": index_offset}}
                cursor = self.db[collection].find(filter).sort(sort_index, sort_order)
            else:
                cursor = self.db[collection].find().sort(sort_index, sort_order)
        else:
            cursor = self.db[collection].find()
        return cursor

    def get_change_stream(
        self,
        collection,
        operation_types: list[str] = ["insert", "update"],
        start_at: int = None,
        max_count: int = None,
    ):
        match_stage = {
            "$match": {
                "operationType": {"$in": operation_types},
            },
        }

        pipeline = [match_stage]

        if start_at is not None:
            start_at_operation_time = bson.timestamp.Timestamp(start_at, 0)
        else:
            start_at_operation_time = None

        try:
            change_stream = self.db[collection].watch(
                pipeline=pipeline, start_at_operation_time=start_at_operation_time
            )
        except pymongo.errors.OperationFailure as e:
            logger.warn(f"× Error: {e}")
            return

        count = 0
        for change in change_stream:
            count += 1
            if max_count and count > max_count:
                break
            logger.mesg(pformat(change, sort_dicts=False, indent=2))
        change_stream.close()


if __name__ == "__main__":
    mongo = MongoOperator()
    start_at_ts = int(datetime.now().timestamp()) - 1 * 60 * 60
    mongo.get_change_stream("videos", start_at=start_at_ts, max_count=1)

    # python -m networks.mongo
