import os
import uuid
import chromadb
from chromadb import PersistentClient
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction                                                    


chroma_client = chromadb.PersistentClient(path="my_vectordb")

embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

collection = chroma_client.get_collection(name="multimodal_collection", 
                                          embedding_function=embedding_function,
                                          data_loader=data_loader)

# Directory containing the images
image_directory = '1950'

# # List all image files in the directory
# image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'))]

# # Generate unique IDs for each image
# ids = [str(uuid.uuid4()) for _ in image_files]

# # Create URIs for each image
# uris = [os.path.join(image_directory, f) for f in image_files]

# # Add images to the collection
# collection.add(
#     ids=ids,
#     uris=uris
# )


# Function to add images to the collection
def add_images_to_collection(image_directory):
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    ids = [str(uuid.uuid4()) for _ in image_files]
    uris = [os.path.join(image_directory, f) for f in image_files]
    
    collection.add(
        ids=ids,
        uris=uris
    )

# Add images to the collection
add_images_to_collection(image_directory)

print(collection.count())

