# Shakespearean Text Generator Using Recurrent Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
- [Overview](#overview)
- [Author](#author)
- [Deep Learning Approach](#deep-learning-approach)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Implementation Details](#implementation-details)
  - [Model Architecture](#model-architecture)
  - [Data Processing](#data-processing)
  - [Training Process](#training-process)
  - [Generation Algorithm](#generation-algorithm)
- [Results](#results)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [License](#license)

## Overview
This repository contains an implementation of a Shakespearean text generator using Long Short-Term Memory (LSTM) networks. The model learns from Shakespeare's complete works (5.7MB corpus) to understand Elizabethan language patterns, vocabulary, and writing style at the word level, then generates new text that captures the poetic and dramatic essence of Shakespeare's writing.

## Author
- Sashwat Desai (desai.sas@northeastern.edu)
  - MS, Applied Mathematics
  - Northeastern University, Boston

## Deep Learning Approach
The implementation uses a word-level LSTM neural network to model sequential patterns in Shakespearean text:
- **LSTM Architecture**: 512-unit LSTM layer to capture long-range dependencies in text sequences
- **Word-Level Modeling**: Processes text at the word level rather than character level for more coherent generation
- **Sequence Learning**: Uses 20-word sequences for training context
- **One-Hot Encoding**: Converts words to boolean vectors for neural network input
- **Temperature Sampling**: Implements controllable randomness in text generation for creative variation

## Key Features
- Complete preprocessing pipeline for cleaning and normalizing historical text
- Word frequency analysis with configurable minimum frequency thresholds
- One-hot encoding for vocabulary of unique words
- Temperature-based sampling for controlling generation creativity
- Model checkpoint saving for best performing weights
- Support for generating variable-length text outputs
- Sample outputs demonstrating Shakespearean language patterns

## Repository Structure
- **code/** - Implementation code
  - **Shakesperean Text Generator.ipynb** - Main Jupyter notebook containing the full implementation
  - **placeholder.md** - Placeholder file
- **data/** - Shakespeare corpus
  - **Shakespeare.txt** - Complete works of Shakespeare (5.7MB)
  - **placeholder.md** - Placeholder file
- **output/** - Generated text samples
  - **sample_output.rtf** - Example generated Shakespearean text
  - **placeholder.md** - Placeholder file
- **.gitignore** - Gitignore file
- **LICENSE** - MIT License
- **README.md** - Project documentation
- **requirements.txt** - Requirements

## Implementation Details

### Model Architecture
The neural network architecture consists of:
- **Input Shape**: (20, n_words) - sequences of 20 words with one-hot encoding
- **LSTM Layer**: 512 units for capturing complex language patterns
- **Output Layer**: Dense layer with softmax activation across vocabulary size
- **Loss Function**: Categorical crossentropy for multi-class classification
- **Optimizer**: Adam with learning rate 0.001
- **Metrics**: Accuracy tracking during training

### Data Processing
The preprocessing pipeline includes:
1. **Text Loading**: UTF-8 encoding for handling special characters
2. **Normalization**: Convert to lowercase for consistency
3. **Character Removal**: Eliminate 50+ special characters and punctuation
4. **Whitespace Handling**: Replace multiple spaces with single space
5. **Word Tokenization**: Split text into individual words
6. **Frequency Filtering**: Keep words appearing more than minimum threshold
7. **Vocabulary Creation**: Build word-to-index and index-to-word mappings

### Training Process
The model training involves:
- Sequence generation using sliding windows of 20 words
- One-hot encoding of input sequences and target words
- Batch size of 128 for efficient GPU utilization
- 100 epochs of training
- ModelCheckpoint callback to save best weights based on loss
- Real-time loss monitoring and model improvement tracking

### Generation Algorithm
The text generation process:
1. **Seed Selection**: Initialize with 20-word sequence from training data
2. **Prediction**: Generate probability distribution over vocabulary
3. **Temperature Sampling**: Apply temperature scaling to control randomness
4. **Word Selection**: Use multinomial sampling to select next word
5. **Sequence Update**: Slide window forward with new word
6. **Iteration**: Continue for specified number of words (600 default)

## Results
- Successfully processes complete Shakespeare corpus (5.7MB)
- Generates coherent Elizabethan-style text
- Maintains Shakespearean vocabulary and phrase structures
- Sample output demonstrates understanding of:
  - Poetic language patterns
  - Archaic word usage
  - Dramatic dialogue style
  - Sonnet-like constructions

## Setup and Installation
```bash
# Clone this repository
git clone https://github.com/desai-sashwat/shakespearean-text-generator.git
cd shakespearean-text-generator

# Install required packages
pip install -r requirements.txt
```

## Usage
The main implementation is in the Jupyter notebook `code/Shakesperean Text Generator.ipynb`. 

### Quick Start
1. Open the notebook:
```bash
jupyter notebook "code/Shakesperean Text Generator.ipynb"
```

2. Run all cells to:
   - Load and preprocess Shakespeare corpus
   - Build vocabulary from processed text
   - Create training sequences
   - Train the LSTM model
   - Generate new Shakespearean text

### Text Generation with Trained Model
```python
# Load saved weights
model.load_weights("s_weights.keras")

# Generate text with different temperature values
# Lower temperature (0.5): More conservative, predictable text
# Higher temperature (1.2): More creative, diverse text
generated_text = generate_text(model, seed_sequence, temperature=1.0, num_words=600)
```

## Future Work
Potential enhancements for this project include:
- Implementing bidirectional LSTMs for better context understanding
- Adding attention mechanisms for improved long-range dependencies
- Character-level modeling for handling rare words and names
- Fine-tuning on specific plays or sonnets for style-specific generation
- Creating a web interface for interactive text generation
- Implementing beam search for higher quality outputs
- Adding rhyme and meter constraints for sonnet generation
- Exploring transformer architectures (GPT-style) for comparison

## License
This project is licensed under the MIT License - see the LICENSE file for details.
matplotlib>=3.5.0
jupyter>=1.0.0
ipython>=7.0.0
