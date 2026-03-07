import argparse
import json
from collections import defaultdict

from bson import BSON
from sedb import ElasticOperator, MongoOperator

from configs.envs import ELASTIC_PRO_ENVS, MONGO_ENVS, SECRETS


def sample_mongo_docs(collection: str, sample_size: int) -> list[dict]:
    mongo = MongoOperator(
        configs=MONGO_ENVS,
        connect_cls=sample_mongo_docs,
        verbose_args=False,
    )
    db = mongo.client[MONGO_ENVS.get("dbname", "bili")]
    return list(db[collection].find({}).sort("_id", 1).limit(sample_size))


def sample_es_docs(index: str, elastic_env: str, sample_size: int) -> list[dict]:
    elastic_envs = SECRETS[elastic_env] if elastic_env else ELASTIC_PRO_ENVS
    es = ElasticOperator(elastic_envs, connect_cls=sample_es_docs)
    res = es.client.search(
        index=index,
        body={
            "query": {"match_all": {}},
            "sort": [{"mid": {"order": "asc"}}],
            "size": sample_size,
        },
    )
    return [
        hit.get("_source") or {} for hit in res.body.get("hits", {}).get("hits", [])
    ]


def summarize_series(values: list[int]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "avg": 0, "p95": 0, "max": 0}
    p95_index = min(max(int(len(ordered) * 0.95) - 1, 0), len(ordered) - 1)
    return {
        "count": len(ordered),
        "avg": round(sum(ordered) / len(ordered), 2),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def summarize_docs(docs: list[dict]) -> dict:
    field_bytes = defaultdict(list)
    doc_sizes = []
    for doc in docs:
        doc_sizes.append(len(BSON.encode(doc)))
        for key, value in doc.items():
            field_bytes[key].append(len(BSON.encode({key: value})))
    return {
        "doc_bytes": summarize_series(doc_sizes),
        "field_bytes": {
            key: summarize_series(values)
            for key, values in sorted(
                field_bytes.items(),
                key=lambda item: sum(item[1]) / max(len(item[1]), 1),
                reverse=True,
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--mongo-collection", required=True)
    parser.add_argument("-i", "--index", required=True)
    parser.add_argument("-ev", "--elastic-env", default="elastic_dev")
    parser.add_argument("-n", "--sample-size", type=int, default=200)
    args = parser.parse_args()

    mongo_docs = sample_mongo_docs(args.mongo_collection, sample_size=args.sample_size)
    es_docs = sample_es_docs(
        args.index, elastic_env=args.elastic_env, sample_size=args.sample_size
    )
    payload = {
        "mongo_collection": args.mongo_collection,
        "elastic_index": args.index,
        "sample_size": args.sample_size,
        "mongo": summarize_docs(mongo_docs),
        "elastic": summarize_docs(es_docs),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
