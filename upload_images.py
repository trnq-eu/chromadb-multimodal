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
def add_images_to_collection(collection, image_directory, batch_size=10):
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
    
    total_images = len(image_files)
    print(f"Found {total_images} images to process...")
    
    # Process images in batches
    for i in tqdm(range(0, total_images, batch_size), desc="Processing image batches", unit="batch"):
        batch_files = image_files[i:i + batch_size]
        batch_ids = []
        batch_uris = []
        
        for f in batch_files:
            batch_ids.append(str(uuid.uuid4()))
            batch_uris.append(os.path.join(image_directory, f))
        
        try:
            # Add the batch to the collection
            collection.add(
                ids=batch_ids,
                uris=batch_uris
            )
        except Exception as e:
            print(f"Error processing batch starting at image {i}: {str(e)}")
            continue
            
    print(f"Finished processing {total_images} images for collection '{collection.name}'.")

# Set up argument parsing
parser = argparse.ArgumentParser(description="Add images to a ChromaDB collection.")
parser.add_argument("--collection_name", type=str, default="multimodal_collection", 
                    help="Name of the collection (default: multimodal_collection)")
parser.add_argument("--db_path", type=str, default="my_vectordb", 
                    help="Path to the database folder (default: my_vectordb)")
parser.add_argument("--batch_size", type=int, default=10,
                    help="Number of images to process in each batch (default: 10)")
parser.add_argument("image_directory", type=str, 
                    help="Path to the directory containing the images")

if __name__ == "__main__":
    # Parse the arguments
    args = parser.parse_args()

    # Create the collection
    collection = create_collection(args.collection_name, args.db_path)

    # Add images to the collection
    add_images_to_collection(collection, args.image_directory, args.batch_size)

    print(f"Collection '{collection.name}' now has {collection.count()} items.")