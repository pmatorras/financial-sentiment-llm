import torch
import gradio as gr
import torch.nn.functional as F
from transformers import AutoTokenizer
from finsentiment.modeling.bert import FinancialSentimentModel

# Use the helper function to get the correct path
CHECKPOINT_PATH = "models/multi_task_model_phase2.pt" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model
print(f"Loading model from {CHECKPOINT_PATH}...")
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = FinancialSentimentModel()

# Load weights (map_location ensures it runs on CPU if CUDA is missing)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

# Prediction Function
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

# Create Interface
demo = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(placeholder="Apple beats earnings estimates...", label="Financial Text"),
    outputs=[
        gr.Label(num_top_classes=3, label="Sentiment Class"), 
        gr.Number(label="FiQA Intensity Score")
    ],
    title="Financial Sentiment LLM",
    description="""
    **Current Status: Phase 2 Validation (Jan 2026)**
    
    This is a research prototype testing a Multi-Task FinBERT architecture.
    It predicts both **Sentiment Class** (Negative/Neutral/Positive) and **Intensity Score** simultaneously.
    
    *Note: This model is actively being developed. Results may vary as we optimize hyperparameters.*
    """,
    examples=[
        ["The company reported a record profit increase of 20%."],
        ["Market uncertainty looms as inflation data disappoints."]
    ]
)

if __name__ == "__main__":
    # Docker requires listening on 0.0.0.0
    demo.launch(server_name="0.0.0.0", server_port=7860)
