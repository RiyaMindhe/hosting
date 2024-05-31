import gradio as gr
from PIL import Image

# Custom function to handle raw image input
def custom_function(image):
    # The input 'image' is a PIL image, no default preprocessing is applied by Gradio
    # You can perform any operation on the raw image here
    # For this example, we'll just return the image itself
    return image

# Create the Gradio interface
iface = gr.Interface(
    fn=custom_function,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil")
)

# Launch the interface
iface.launch()

