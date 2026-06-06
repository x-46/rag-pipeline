from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "rag_collection"
    
    embedding_model_base_url: str
    embedding_model_name: str
    embedding_model_api_key: str
    
    llm_model_base_url: str
    llm_model_name: str
    llm_model_api_key: str

    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "rag"

    class Config:
        env_file = ".env"


settings = Settings()
