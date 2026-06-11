from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection

from rag_core.config import settings

if TYPE_CHECKING:
    from rag_core.metrics import RequestMetrics

_METRICS_COLLECTION = "request_metrics"


def _get_client() -> MongoClient:
    return MongoClient(settings.mongo_uri)



def get_mongo_collection() -> Collection:
	"""Return the Mongo collection used for parent elements."""
	client = _get_client()
	db = client[settings.mongo_db]
	return db[settings.collection_name]


def recreate_mongo_collection_with_parent_elements(parent_elements: list[dict]) -> Collection:
	"""Drop and recreate Mongo collection, then upsert parent elements by element_id."""
	client = _get_client()
	db = client[settings.mongo_db]

	db.drop_collection(settings.collection_name)
	mongo_collection = db[settings.collection_name]
	mongo_collection.create_index("element_id", name="idx_element_id")

	ops: list[UpdateOne] = []
	for item in parent_elements:
		doc = dict(item)
		element_id = doc.get("element_id")

		if element_id:
			ops.append(
				UpdateOne(
					{"element_id": element_id},
					{"$set": doc},
					upsert=True,
				)
			)
		else:
			print("Warning: Skipping document without element_id")

	if ops:
		result = mongo_collection.bulk_write(ops, ordered=False)
		print(
			f"Upserted: {result.upserted_count}, "
			f"Modified: {result.modified_count}, "
			f"Matched: {result.matched_count}"
		)

	print(f"MongoDB collection ready: {settings.mongo_db}.{settings.collection_name}")
	print("Index info:", mongo_collection.index_information())
	return mongo_collection


def save_request_metrics(metrics: RequestMetrics, question: str) -> None:
    """Persist a RequestMetrics snapshot for one request to the request_metrics collection."""
    def _phase(p) -> dict:
        return {
            "llm_input_tokens": p.llm_input_tokens,
            "llm_output_tokens": p.llm_output_tokens,
            "llm_ms": round(p.llm_ms, 2),
            "db_ms": round(p.db_ms, 2),
            "local_ms": round(p.local_ms, 2),
        }

    total = metrics._total()

    doc = {
        "timestamp": datetime.now(timezone.utc),
        "question": question,
        "phases": {
            "query_rewrite": _phase(metrics.query_rewrite),
            "retrieval":     _phase(metrics.retrieval),
            "parent_fetch":  _phase(metrics.parent_fetch),
            "answer":        _phase(metrics.answer),
        },
        "totals": {
            **_phase(total),
            "wall_ms": round(metrics.wall_ms, 2),
        },
    }

    client = _get_client()
    db = client[settings.mongo_db]
    db[_METRICS_COLLECTION].insert_one(doc)
