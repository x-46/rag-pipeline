from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection

from rag_core.config import settings


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


