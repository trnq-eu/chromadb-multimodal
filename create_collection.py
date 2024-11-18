import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

# create a new chroma collection
def create_collection(collection_name="multimodal_collection", db_path="my_vectordb"):
    chroma_client = chromadb.PersistentClient(path=db_path)

    embedding_function = OpenCLIPEmbeddingFunction()
    data_loader = ImageLoader()

    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        data_loader=data_loader)


# Create the collection
create_collection()
