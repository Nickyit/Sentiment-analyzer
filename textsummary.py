import torch
from transformers import pipeline

model_path = r"C:\Users\harsh\.cache\huggingface\hub\models--facebook--bart-large-cnn\snapshots\37f520fa929c961707657b28798b30c003dd100b"

summarizer = pipeline(
    "summarization",
    model=model_path,

    dtype=torch.float32
)

# text = """Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and 
#         entrepreneur known for his leadership of Tesla, SpaceX, Twitter, and xAI. Musk has
#          been the wealthiest person in the world since 2021; as of December 2025, Forbes
#         estimates his net worth to be around US$717 billion..."""

# result = summarizer(
#     text,
#     max_new_tokens=150,
#     truncation=True,
#     do_sample=False
# )

# print(result[0]["summary_text"])
# print(summarizer.model.config)
# print(summarizer(text, max_new_tokens=100, truncation=True, do_sample=False))

# importing gradio to close any existing interfaces
import gradio as gr

gr.close_all()

# We create a function bcs gradio needs a function to work with
def summary(text):
    output = summarizer(text)
    return output[0]["summary_text"]

interface = gr.Interface(
    fn=summary,
    inputs=gr.Textbox(
        label="Input Text",
        placeholder="Paste your long text here...",
        lines=8
    ),
    outputs=gr.Textbox(
        label="Summarized Text",
        placeholder="Summary will appear here...",
        lines=8
    ),
    title="Text Summarization App",
    description="Paste a long paragraph and get a short summary using BART Large CNN model.",
    allow_flagging="never"
)

interface.launch()

