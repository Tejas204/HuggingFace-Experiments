from transformers import pipeline

# Passing one sentence to pipeline
classifier = pipeline("sentiment-analysis")
print(classifier("I havebeen waiting for hugging face course my whole life"))

# ------------------------------------------------------------------------------------------------------------------------
# The most basic object in the 🤗 Transformers library is the pipeline() function. It connects a model with its necessary 
# preprocessing and postprocessing steps, allowing us to directly input any text and get an intelligible answer.
#
# By default, this pipeline selects a particular pretrained model that has been fine-tuned for sentiment analysis in English. 
# The model is downloaded and cached when you create the classifier object. If you rerun the command, the cached model 
# will be used instead and there is no need to download the model again.
#
# Passing several sentences to the pipeline
#------------------------------------------------------------------------------------------------------------------------
print(classifier(["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]))