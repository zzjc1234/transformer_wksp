# GPT Model Implementation from Scratch - Source Documentation

This directory contains a complete implementation of a GPT model built from scratch based on the book "Build a Large Language Model From Scratch" by Sebastian Raschka. Each file implements a specific aspect of the model architecture and functionality.

## File Descriptions

### Core Model Implementation
- **`gpt_model_from_reference.py`** - Complete GPT model implementation containing all components from the book:
  - `GPTDatasetV1`: Dataset class for preparing training data
  - `create_dataloader_v1`: Function to create data loaders for training
  - `MultiHeadAttention`: Multi-head attention mechanism implementation
  - `LayerNorm`: Layer normalization module
  - `GELU`: Gaussian Error Linear Unit activation function
  - `FeedForward`: Position-wise feed-forward network
  - `TransformerBlock`: Complete transformer block with attention and FFN
  - `GPTModel`: Main GPT model class combining all components
  - `generate_text_simple`: Text generation function for autoregressive generation
  - **Usage**: Contains the main GPT model architecture and text generation capabilities

- **`gpt_with_kv_cache.py`** - GPT model implementation with KV (Key-Value) caching for efficient inference:
  - Implements caching of Key and Value matrices to speed up generation
  - Reduces computational complexity from O(t²) to O(t) per token during generation
  - Includes optimized text generation with caching functionality
  - **Usage**: Use for efficient text generation where computation needs to be optimized

- **`gpt_with_kv_cache_optimized.py`** - Optimized version of KV caching implementation:
  - Enhanced KV caching with additional optimizations
  - More memory-efficient caching mechanisms
  - **Usage**: Use for production deployment where performance is critical

### Data Processing
- **`data_module.py`** - Partial implementation of data processing components
  - Contains `GPTDatasetV1` class for dataset handling
  - **Usage**: For understanding dataset creation (refer to full implementation in gpt_model_from_reference.py)

### Attention Mechanisms
- **`attention_implementation.ipynb`** - Jupyter notebook with detailed attention mechanism implementation:
  - Step-by-step implementation of scaled dot-product attention
  - Multi-head attention mechanism with visualizations
  - Mathematical explanations and code examples
  - **Usage**: Educational purposes to understand attention mechanisms in detail

### Utilities and Demos
- **`demo.py`** - Demonstration script showcasing GPT model functionality:
  - Creates and initializes a GPT model
  - Performs text generation with sample input
  - Displays model configuration and parameters
  - **Usage**: Run `python demo.py` to see the model in action

- **`__init__.py`** - Makes the directory a Python package
  - **Usage**: Enables importing modules from this directory

### Additional Implementations
- **`gpt_complete.py`**, **`gpt_full_implementation.py`**, **`gpt_full.py`**, **`gpt_model_complete.py`**, **`gpt_model.py`**, **`gpt_part1.py`** - Partial implementations created during development
  - These contain various attempts at implementing different components
  - **Usage**: For reference only; use `gpt_model_from_reference.py` for complete implementation

- **`data_processing.py`**, **`dataset.py`** - Additional data processing implementations
  - **Usage**: For reference only; main implementation is in `gpt_model_from_reference.py`

## Running the Code

### Prerequisites
Ensure you have the required dependencies installed. From the project root directory:

```bash
uv add tiktoken torch matplotlib
```

### Running the Demo
```bash
cd /Users/zhouzhaojiacheng/research/unigroup/transformer_wksp
source .venv/bin/activate
python src/demo.py
```

### Using the Model in Your Own Code
```python
from src.gpt_model_from_reference import GPTModel, generate_text_simple

# Configure model parameters
GPT_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 256,   # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}

# Create the model
model = GPTModel(GPT_CONFIG)

# Generate text
# (Follow the example in demo.py for full usage)
```

## Key Features Implemented

1. **GPT Dataset Creation**: Efficient dataset class for preparing training data
2. **Multi-Head Attention**: Complete attention mechanism with causal masking
3. **Transformer Architecture**: Complete transformer block with normalization and FFN
4. **Text Generation**: Autoregressive text generation capabilities
5. **KV Caching**: Optimized inference with key-value caching
6. **Configurable Architecture**: Easily adjustable model parameters

## Architecture Overview

The GPT model follows the decoder-only transformer architecture:
- Token embeddings + Positional embeddings
- Multiple transformer blocks with self-attention
- Each transformer block contains:
  - Multi-head self-attention mechanism
  - Position-wise feed-forward network
  - Layer normalization and residual connections
- Final normalization and output projection

## Dependencies
- torch: Deep learning framework
- tiktoken: Tokenization library (for GPT-2 tokenizer)
- matplotlib: For plotting utilities (if needed)
- Jupyter: For notebook execution

## Note on NumPy Warnings
You may see warnings about NumPy version compatibility. These are due to version mismatches between different packages but do not affect the functionality of the code.
