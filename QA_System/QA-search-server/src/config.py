import os

MILVUS_HOST = os.getenv("MILVUS_HOST", "10.33.43.39")
MILVUS_PORT = os.getenv("MILVUS_PORT", 19530)

PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = os.getenv("PG_PORT", 5432)
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_DATABASE = os.getenv("PG_DATABASE", "postgres")
MONGO_HOST = os.getenv("MONGO_HOST", "10.33.43.39")
MONGO_PORT = os.getenv("MONGO_PORT", 27017)
BERT_CLIENT_HOST = os.getenv("BERT_CLIENT_HOST", "10.33.43.27")

DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "medical_qa")
