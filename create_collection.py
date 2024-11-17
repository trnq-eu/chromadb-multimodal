# file to create a chroma collection

import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction


chroma_client = chromadb.PersistentClient(path="my_vectordb")

embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

collection = chroma_client.create_collection(
    name='multimodal_collection',
    embedding_function=embedding_function,
    data_loader=data_loader)