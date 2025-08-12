# Use a pipeline as a high-level helper
from transformers import pipeline

# Set a local directory inside your project
model_dir = "./model"

# Download & use the model from that directory
classifier = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

print(classifier("I hate books!"))