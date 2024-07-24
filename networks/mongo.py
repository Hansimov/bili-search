import pymongo

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


if __name__ == "__main__":
    syncer = MongoSyncer()
    syncer.watch()

    # python -m elastics.mongo_syncer
