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
def load_image(image):
    if isinstance(image, str):
        # If image is a file path
        image = Image.open(image)
    elif hasattr(image, 'name'):
        # If image is a temporary file from Gradio
        image = Image.open(image.name)
    else:
        # If image is already a PIL Image
        return image
    
    image = image.convert('RGB')  # Ensure the image is in RGB format
    return image

# Function to query the collection with a text query
def query_text(text_query, n_results=2):
    try:
        results = collection.query(
            query_texts=[text_query],  # Changed from query_text to query_texts and wrap in list
            n_results=n_results,
            include=['documents', 'distances', 'metadatas', 'uris']
        )
        return results
    except Exception as e:
        print(f"Error in text query: {e}")
        return None

# Function to query the collection with an uploaded image
def query_image(uploaded_image, n_results=2):
    try:
        # Process the uploaded image
        query_image = load_image(uploaded_image)
        
        # Convert PIL Image to numpy array
        query_array = np.array(query_image)
        
        # Query the collection using the embedding
        results = collection.query(
            query_images=[query_array],  # Wrap in list
            n_results=n_results,
            include=['documents', 'distances', 'metadatas', 'uris']
        )
        
        return results
    except Exception as e:
        print(f"Error in image query: {e}")
        return None

# Function to process and display the results as a gallery
def process_results(query_results):
    if not query_results or 'ids' not in query_results or not query_results['ids']:
        return []
    
    result_count = len(query_results['ids'][0])
    gallery_images = []
    
    for j in range(result_count):
        try:
            id = query_results["ids"][0][j]
            distance = query_results['distances'][0][j]
            uri = query_results['uris'][0][j] if 'uris' in query_results else None
            
            if uri and os.path.exists(uri):
                # Load and prepare the image for display
                image = load_image(uri)
                gallery_images.append((image, f'ID: {id}, Distance: {distance:.4f}'))
        except Exception as e:
            print(f"Error processing result {j}: {e}")
            continue
    
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
        
        if query_results is None:
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
            placeholder="Enter text query here...",
            visible=True
        )
        image_input = gr.Image(
            type="filepath",  # Changed from 'pil' to 'filepath'
            label="Image Query",
            visible=False
        )
    
    with gr.Row():
        n_results = gr.Slider(
            minimum=1,
            maximum=10,
            step=1,
            value=2,
            label="Number of Results"
        )

    # Set up the submit button
    submit_btn = gr.Button("Search")
    
    gallery_output = gr.Gallery(label="Similar Images")
    
    # Set up event handlers
    def update_visibility(query_type):
        return {
            text_input: gr.update(visible=query_type=="Text Query"),
            image_input: gr.update(visible=query_type=="Image Query")
        }
    
    query_type.change(
        fn=update_visibility,
        inputs=[query_type],
        outputs=[text_input, image_input]
    )
    
    
    submit_btn.click(
        fn=gradio_interface,
        inputs=[query_type, text_input, image_input, n_results],
        outputs=gallery_output
    )

# Launch the interface
iface.launch(share=True)