import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

chroma_client = chromadb.PersistentClient(path="my_vectordb")


def delete_collection(collection_name):
    chroma_client.delete_collection(name=collection_name)

delete_collection('multimodal_collection')