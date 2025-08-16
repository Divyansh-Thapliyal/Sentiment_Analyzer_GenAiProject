# Use a pipeline as a high-level helper
from transformers import pipeline
import gradio as gr

# Set a local directory inside your project
model_dir = "./model"

# Download & use the model from that directory
classifier = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# print(classifier("I hate books!"))



# Now lets create ui using gradio. So lets create a function.

def sentiment_analysis(text):
    result = classifier(text)
    return result


demo = gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox(lines=5, label="Input", placeholder="Enter text here..."),
    outputs=gr.Textbox(lines=5, label="Result",placeholder="Sentiment result will appear here..."),
    title= "Sentiment Analysis Project",
    description="This is a simple sentiment analysis project using Hugging Face Transformers and Gradio.",
    theme="default"

)

# Launch the Gradio app
demo.launch()
