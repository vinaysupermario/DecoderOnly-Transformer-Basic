import torch
import gradio as gr
from transformers import GPT2Tokenizer

from transformer import DecoderOnlyTransformer, Config  # Custom Model Code

# Load model and tokenizer
def load_model():
    # Load config
    config = Config(vocab_size=50257)  # Adjust to match your config
    
    # Initialize model
    model = DecoderOnlyTransformer(config)
    checkpoint = torch.load("model/decoder_transformer.pth", map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("model/tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

model, tokenizer = load_model()

# Text generation function
def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(  # Implement a generate method in your model
            input_ids, 
            max_length=max_length, 
            temperature=0.7
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Gradio interface
demo = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=2, placeholder="Enter prompt..."),
    outputs=gr.Textbox(label="Generated Text"),
    examples=[
        ["Once upon a time, there was a"],
        ["In a futuristic city, AI robots"],
        ["The secret to happiness is"],
        ["Deep learning is"],
        ["When I opened the door, I saw"]
    ],
    title="Decoder-Only Transformer Demo",
    description="Generate text with my custom transformer model!"
)

demo.launch()
