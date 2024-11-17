import os
import uuid
import numpy as np
from PIL import Image
import chromadb
from chromadb import PersistentClient
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
import gradio as gr

# Initialize ChromaDB client and collection
chroma_client = PersistentClient(path="my_vectordb")
embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

collection = chroma_client.get_collection(
    name="multimodal_collection",
    embedding_function=embedding_function,
    data_loader=data_loader
)

# Function to load and preprocess an image
def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')  # Ensure the image is in RGB format
    return image

# Function to query the collection with a text query
def query_text(text_query, n_results=2):
    query_results = collection.query(
        query_texts=[text_query],
        n_results=n_results,
        include=['documents', 'distances', 'data', 'uris', 'metadatas'],
    )
    return query_results

# Function to query the collection with an uploaded image
def query_image(uploaded_image, n_results=2):
    # Load the uploaded image
    if isinstance(uploaded_image, str):
        uploaded_image = load_image(uploaded_image)
    
    # Generate the embedding for the uploaded image
    uploaded_image_embedding = embedding_function([uploaded_image])
    
    # Query the collection using the embedding
    query_results = collection.query(
        query_embeddings=uploaded_image_embedding,
        n_results=n_results,
        include=['documents', 'distances', 'data', 'uris', 'metadatas'],
    )
    
    return query_results

# Function to process and display the results as a gallery
def process_results(query_results):
    result_count = len(query_results['ids'][0]) if query_results['ids'] else 0
    gallery_images = []
    for j in range(result_count):
        id = query_results["ids"][0][j]
        distance = query_results['distances'][0][j]
        uri = query_results['uris'][0][j]
        
        # Load and prepare the image for display
        image = load_image(uri)
        gallery_images.append((image, f'ID: {id}, Distance: {distance:.4f}'))
    
    return gallery_images

# Modified Gradio Interface function to handle all inputs
def gradio_interface(query_type, text_query, image_query, n_results):
    try:
        if query_type == "Text Query" and text_query:
            query_results = query_text(text_query, int(n_results))
        elif query_type == "Image Query" and image_query is not None:
            query_results = query_image(image_query, int(n_results))
        else:
            return []
        
        return process_results(query_results)
    except Exception as e:
        print(f"Error in gradio_interface: {e}")
        return []

# Define the Gradio interface with conditional inputs
with gr.Blocks() as iface:
    gr.Markdown("# ChromaDB Image Search")
    gr.Markdown("Upload an image or enter a text query to find similar images in the collection.")
    
    with gr.Row():
        query_type = gr.Radio(
            choices=["Text Query", "Image Query"],
            label="Query Type",
            value="Text Query"
        )
    
    with gr.Row():
        text_input = gr.Textbox(
            label="Text Query",
            lines=1,
            placeholder="Enter text query here..."
        )
        image_input = gr.Image(
            type="pil",
            label="Image Query"
        )
    
    with gr.Row():
        n_results = gr.Slider(
            minimum=1,
            maximum=10,
            step=1,
            value=2,
            label="Number of Results"
        )
    
    gallery_output = gr.Gallery(label="Similar Images")
    
    # Set up event handlers
    query_type.change(
        fn=lambda x: [gr.update(visible=(x=="Text Query")), gr.update(visible=(x=="Image Query"))],
        inputs=[query_type],
        outputs=[text_input, image_input]
    )
    
    # Set up the main query function
    components = [query_type, text_input, image_input, n_results]
    for component in [text_input, image_input, n_results]:
        component.change(
            fn=gradio_interface,
            inputs=components,
            outputs=gallery_output
        )

# Launch the interface
iface.launch(debug=True)