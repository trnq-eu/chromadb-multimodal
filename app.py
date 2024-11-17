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

collection.add(
    ids=['0','1'],
    uris=['/root/Projects/Images/1950/1280px-blossom_restaurant_103_bowery_by_berenice_abbott_in_1935.webp', 
         '/root/Projects/Images/1950/1950s_Afghanistan_-_Textile_store_window_display.jpg']

)