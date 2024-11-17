import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from matplotlib import pyplot as plt


chroma_client = chromadb.PersistentClient(path="my_vectordb")

embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

# collection = chroma_client.create_collection(
#     name='multimodal_collection',
#     embedding_function=embedding_function,
#     data_loader=data_loader)

collection = chroma_client.get_collection(name="multimodal_collection", 
                                          embedding_function=embedding_function,
                                          data_loader=data_loader)

data_loader = ImageLoader()


collection.add(
    ids=['0','1'],
    uris=['1950/1950S-BANGKOK-STREET-SCENE.jpg', 
         '1950/Christmas_party_in_1950s_New_Orleans.jpg']

)

# Simple function to print the results of a query.
# The 'results' is a dict {ids, distances, data, ...}
# Each item in the dict is a 2d list.
def print_query_results(query_list: list, query_results: dict)->None:
    result_count = len(query_results['ids'][0])

    for i in range(len(query_list)):
        print(f'Results for query: {query_list[i]}')

        for j in range(result_count):
            id       = query_results["ids"][i][j]
            distance = query_results['distances'][i][j]
            data     = query_results['data'][i][j]
            document = query_results['documents'][i][j]
            # metadata = query_results['metadatas'][i][j]
            uri      = query_results['uris'][i][j]

            # print(f'id: {id}, distance: {distance}, metadata: {metadata}, document: {document}') 
            print(f'id: {id}, distance: {distance}, document: {document}') 

            # Display image, the physical file must exist at URI.
            # (ImageLoader loads the image from file)
            print(f'data: {uri}')
            plt.imshow(data)
            plt.axis("off")
            plt.show()


query_texts = ['building']

query_results = collection.query(
    query_texts = query_texts,
    n_results = 2,
    include=['documents', 'distances', 'data', 'uris'],
)

print_query_results(query_texts, query_results)

# print(collection.count())

