import torch
import gradio as gr
import torch.nn.functional as F
from transformers import AutoTokenizer

# CORRECTED IMPORTS for src-layout
from finsentiment.modeling.bert import FinancialSentimentModel
from finsentiment.config import MODEL_NAME, get_model_path

# 1. Configuration
# We use the helper function from your config to get the correct path
CHECKPOINT_PATH = "models/multi_task_model.pt" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load Model
print(f"Loading model from {CHECKPOINT_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = FinancialSentimentModel()
# Load weights (map_location ensures it runs on CPU if CUDA is missing)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 3. Prediction Function
def analyze_text(text):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=128
    ).to(DEVICE)
    
    with torch.no_grad():
        # Head 1: Classification
        cls_logits = model(inputs["input_ids"], inputs["attention_mask"], task_type="classification")
        probs = F.softmax(cls_logits, dim=1).squeeze().cpu().tolist()
        
        # Head 2: Regression
        reg_score = model(inputs["input_ids"], inputs["attention_mask"], task_type="regression")
        score = reg_score.item()

    # Format Output
    labels = ["Negative", "Neutral", "Positive"]
    # Handle single-sample batch dimension issues if they arise
    if isinstance(probs, float): probs = [probs] # Edge case for single class
    
    cls_output = {label: prob for label, prob in zip(labels, probs)}
    
    return cls_output, score

# 4. Interface
demo = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(placeholder="Apple beats earnings estimates...", label="Financial Text"),
    outputs=[
        gr.Label(num_top_classes=3, label="Sentiment Class"), 
        gr.Number(label="FiQA Intensity Score")
    ],
    title="Financial Sentiment LLM",
    description="Multi-task FinBERT: Classification + Regression.",
    examples=[
        ["The company reported a record profit increase of 20%."],
        ["Market uncertainty looms as inflation data disappoints."]
    ]
)

if __name__ == "__main__":
    # Docker requires listening on 0.0.0.0
    demo.launch(server_name="0.0.0.0", server_port=7860)
