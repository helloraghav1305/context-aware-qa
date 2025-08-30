import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"
MODEL_WEIGHTS_PATH = "model_weights.pth"
MODEL_SAVE_PATH = "model"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)

# Load model + weights
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()

def generate_reasoning_answer(question, context):
    input_text = f"answer the question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_length=512,
        num_beams=4,
        early_stopping=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

examples = [
    [
        "Who brought the snacks to the park?",
        "Karan, Rohan, and Ravi went to the park on Saturday afternoon. Karan brought his football, and Rohan brought some snacks to share. Ravi forgot to bring his water bottle, so he borrowed Karan's bottle during their break. After playing for two hours, they all sat under a tree and ate the snacks that Rohan had packed."
    ],
    [
        "Why does Ravi like spending time with his plants?",
        "Ravi is a college student who is studying Computer Science. Every evening, he goes to the terrace to water his small garden. He has planted tomatoes and chillies. Last week, he harvested fresh tomatoes and made a delicious salad for dinner. He loves spending time with his plants because it makes him feel peaceful after a busy day of classes and assignments."
    ]
]

# Gradio interface
iface = gr.Interface(
    fn=generate_reasoning_answer,
    inputs=[
        gr.Textbox(lines=2, label="Question", placeholder="Ask your question here..."),
        gr.Textbox(lines=10, label="Context", placeholder="Paste relevant context or document text here...")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Conversational QA",
    description="Ask a question and provide the relevant context. The model will generate an answer based on the context.",
    examples=examples
)

iface.launch()
