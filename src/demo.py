"""
Demonstration script for the GPT model implementation
Based on the 'Build a Large Language Model From Scratch' book by Sebastian Raschka
"""

import tiktoken
import torch
from gpt_model_from_reference import GPTModel, generate_text_simple


def main():
    print("GPT Model Implementation from Scratch - Demo")
    print("=" * 50)

    # Configuration for a small GPT model
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 256,  # Context length (reduced for demo)
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-Key-Value bias
    }

    print("Creating GPT model with configuration:")
    for key, value in GPT_CONFIG_124M.items():
        print(f"  {key}: {value}")

    print("\nInitializing model...")
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # disable dropout

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Sample text generation
    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\nInput text: '{start_context}'")
    print(f"Encoded input: {encoded}")
    print(f"Tensor shape: {encoded_tensor.shape}")

    print("\nGenerating text...")
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
    )

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(f"Generated text: '{decoded_text}'")

    print("\nDemo completed successfully!")
    print("\nKey components implemented:")
    print("- GPT Dataset for training data preparation")
    print("- Multi-Head Attention mechanism")
    print("- Transformer Blocks with LayerNorm and Feed-Forward")
    print("- KV caching for efficient inference (in separate files)")
    print("- Text generation utilities")


if __name__ == "__main__":
    main()
