import os
import uuid
import argparse
import chromadb
from chromadb import PersistentClient
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from tqdm import tqdm

# Function to create or get a collection
def create_collection(collection_name="multimodal_collection", db_path="my_vectordb"):
    chroma_client = PersistentClient(path=db_path)

    embedding_function = OpenCLIPEmbeddingFunction()
    data_loader = ImageLoader()

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        data_loader=data_loader
    )
    
    return collection

# Function to add images to the collection with a progress bar
def add_images_to_collection(collection, image_directory):
    if not os.path.isdir(image_directory):
        print(f"Error: {image_directory} is not a valid directory.")
        return
    
    image_files = [
        f for f in os.listdir(image_directory)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'))
    ]
    
    if not image_files:
        print(f"Warning: No image files found in {image_directory}.")
        return
    
    ids = []
    uris = []
    
    # Use tqdm to create a progress bar
    for f in tqdm(image_files, desc=f"Adding images to collection '{collection.name}'", unit="image"):
        ids.append(str(uuid.uuid4()))
        uris.append(os.path.join(image_directory, f))
    
    collection.add(
        ids=ids,
        uris=uris
    )
    print(f"Added {len(image_files)} images to the collection '{collection.name}'.")

# Set up argument parsing
parser = argparse.ArgumentParser(description="Add images to a ChromaDB collection.")
parser.add_argument("--collection_name", type=str, default="multimodal_collection", help="Name of the collection (default: ffm_collection)")
parser.add_argument("--db_path", type=str, default="my_vectordb", help="Path to the database folder (default: my_vectordb)")
parser.add_argument("image_directory", type=str, help="Path to the directory containing the images")

if __name__ == "__main__":
    # Parse the arguments
    args = parser.parse_args()

    # Create the collection
    collection = create_collection(args.collection_name, args.db_path)

    # Add images to the collection
    add_images_to_collection(collection, args.image_directory)

    print(f"Collection '{collection.name}' now has {collection.count()} items.")

