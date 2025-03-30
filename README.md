# ARC-AGI Dataset Loader

This project provides a PyTorch dataset loader for the Abstraction and Reasoning Corpus (ARC) datasets. It aims to be a simple, easy-to-use starting point for working with different ARC datasets, allowing researchers to quickly experiment with these challenging reasoning tasks.

It supports loading data from:

1. **ARC-AGI v1**: The original ARC dataset from François Chollet's [ARC-AGI Repository](https://github.com/fchollet/ARC-AGI) 
2. **ARC-AGI v2**: The updated ARC dataset from the [ARC Prize Repository](https://github.com/arcprize/ARC-AGI-2)
3. **RE-ARC**: Procedurally generated variations of ARC v1 tasks from [Michael Hodel's RE-ARC Repository](https://github.com/michaelhodel/re-arc)

The implementation is specifically optimized for training Language Models (LLMs) on ARC tasks.

## Dataset Structures

### ARC-AGI (v1)
- **Training Set**: 400 tasks for training
- **Evaluation Set**: 400 tasks for validation
- **License**: Apache-2.0

### ARC-AGI-2 (v2)
- **Training Set**: 1000 tasks for training
- **Evaluation Set**: 120 tasks for validation
- **License**: Apache-2.0

### RE-ARC
- **Dataset**: ~400 task files, each containing ~1000 procedurally generated examples of ARC tasks
- **Structure**: The data is split 80/20 for training and evaluation (approximately 320 tasks for training)
- **License**: MIT

Each task contains:
- Training examples: Input/output pairs demonstrating the pattern
- Test examples: Inputs for which the model must predict the output

## Tokenization

The implementation uses specialized tokenization for ARC grids. Note that this is just an example tokenization implementation - you can adapt it or replace it with your own approach as needed.

- **Input vocabulary**: `<pad>`, `<bos>`, `<eos>`, `\n`, and digits 0-9
- **Output vocabulary**: `<pad>`, `<eos>`, `\n`, and digits 0-9

The grid structure is preserved using:
- Newline token (`\n`) to separate rows
- Individual tokens for each color (0-9)

Example grid and its tokenization:
```
Grid:           Tokens (Input):           Tokens (Output):
[0, 1]          <bos>, 0, 1, \n, 2, 3,    0, 1, \n, 2, 3, <eos>
[2, 3]          <eos>
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Setup

You need to download the datasets and place them in the appropriate folders to get started. Here's the expected folder structure:

```
ARC-AGI-Dataset-Loader/
├── arc_v1_data/              # ARC-AGI v1 data
│   ├── arc-agi_training_challenges.json
│   ├── arc-agi_training_solutions.json
│   ├── arc-agi_evaluation_challenges.json
│   └── arc-agi_evaluation_solutions.json
├── arc_v2_data/              # ARC-AGI-2 data
│   ├── training/             # Training tasks (individual JSON files)
│   │   ├── task1.json
│   │   ├── task2.json
│   │   └── ... (1000 task files)
│   └── evaluation/           # Evaluation tasks (individual JSON files)
│       ├── task1.json
│       ├── task2.json
│       └── ... (120 task files)
├── re_arc_data/              # RE-ARC data
│   └── tasks/                # Contains task JSON files
│       ├── task1.json
│       ├── task2.json
│       └── ...
└── ...
```

Download the datasets from their respective repositories:
- [ARC-AGI v1 Repository](https://github.com/fchollet/ARC-AGI)
- [ARC-AGI-2 v2 Repository](https://github.com/arcprize/ARC-AGI-2)
- [RE-ARC Repository](https://github.com/michaelhodel/re-arc)

## Configuration

All necessary parameters can be set in the `config.py` file, including:
- Data paths
- Tokenizer configuration
- Training parameters
- Dataset settings

This centralized configuration makes it easy to adjust settings without modifying code.

## Usage

### Basic Usage

#### ARC-AGI (v1)

```python
from arc_dataset import ARCV1LLMDataset, ARCV1TaskDataset, create_arc_v1_dataloader
from config import ARC_V1_DATA_PATH

# Dataset for training (individual examples)
train_dataset = ARCV1LLMDataset(ARC_V1_DATA_PATH, split="training")

# Retrieve a specific task by its ID
task_dataset = ARCV1TaskDataset(ARC_V1_DATA_PATH, split="training")
task = task_dataset.get_task_by_id("007bbfb7")

# Create DataLoader for training
train_dataloader = create_arc_v1_dataloader(
    data_path=ARC_V1_DATA_PATH,
    split="training",
    batch_size=32,
    shuffle=True
)
```

#### ARC-AGI-2 (v2)

```python
from arc_dataset import ARCV2LLMDataset, ARCV2TaskDataset, create_arc_v2_dataloader
from config import ARC_V2_DATA_PATH

# Dataset for training (individual examples)
train_dataset = ARCV2LLMDataset(ARC_V2_DATA_PATH, split="training")

# Retrieve a specific task by its ID
task_dataset = ARCV2TaskDataset(ARC_V2_DATA_PATH, split="training")
task = task_dataset.get_task_by_id("00576224")  # Example ID

# Create DataLoader for training
train_dataloader = create_arc_v2_dataloader(
    data_path=ARC_V2_DATA_PATH,
    split="training",
    batch_size=32,
    shuffle=True
)

# DataLoader for entire tasks (with custom collate function)
task_dataloader = create_arc_v2_dataloader(
    data_path=ARC_V2_DATA_PATH,
    split="training",
    batch_size=4,
    shuffle=True,
    by_task=True  # Important: Loads complete tasks instead of individual examples
)
```

#### RE-ARC Dataset

```python
from arc_dataset import REARCLLMDataset, REARCTaskDataset, create_re_arc_dataloader
from config import RE_ARC_DATA_PATH

# Dataset for training (individual examples)
train_dataset = REARCLLMDataset(RE_ARC_DATA_PATH, split="training")

# Dataset for entire tasks with limited examples per task
task_dataset = REARCTaskDataset(RE_ARC_DATA_PATH, split="training", examples_per_task=10)

# Create DataLoader for training individual examples
train_dataloader = create_re_arc_dataloader(
    data_path=RE_ARC_DATA_PATH,
    split="training",
    batch_size=32,
    shuffle=True,
    by_task=False
)

# Create DataLoader for entire tasks
task_dataloader = create_re_arc_dataloader(
    data_path=RE_ARC_DATA_PATH,
    split="training",
    batch_size=4,
    shuffle=True,
    by_task=True,
    examples_per_task=10  # Limits the number of examples per task to prevent memory issues
)
```

### Dataset Classes

This implementation provides two types of datasets for each ARC version:

#### LLM Datasets
- `ARCV1LLMDataset`, `ARCV2LLMDataset`, `REARCLLMDataset`
- **Purpose**: Optimized for standard sequence-to-sequence training
- **Returns**: Individual examples (single input/output grid pairs)
- **Output Format**: Each datapoint is a dictionary with `input_ids`, `output_ids`, `is_test` flag, and optionally `task_id`
- **When to Use**: When training standard LLMs on individual grid transformations
- **DataLoader Flag**: `by_task=False` (default)

#### Task Datasets
- `ARCV1TaskDataset`, `ARCV2TaskDataset`, `REARCTaskDataset`
- **Purpose**: Preserves task structure for few-shot learning, meta-learning, or task-based approaches
- **Returns**: Complete tasks with all training and test examples grouped together
- **Output Format**: Each datapoint is a dictionary with:
  - `task_id`: The unique identifier for the task
  - `train`: A list of all training examples for this task
  - `test`: A list of all test examples for this task
- **When to Use**: When your model needs to see all examples of a task together
- **DataLoader Flag**: `by_task=True`

### DataLoader Configuration

All `create_*_dataloader` functions support two fundamental modes controlled by the `by_task` parameter:

```python
# Individual examples mode (LLMDataset) - Default
dataloader = create_arc_v1_dataloader(
    data_path=ARC_V1_DATA_PATH,
    by_task=False,  # Default setting
    batch_size=32   # Each batch contains 32 individual examples
)

# Task-based mode (TaskDataset)
dataloader = create_arc_v1_dataloader(
    data_path=ARC_V1_DATA_PATH,
    by_task=True,   # Enable task-based loading
    batch_size=4    # Each batch contains 4 complete tasks
)
```

#### What You Get in Each Mode:

##### With `by_task=False` (Default):
- Uses the corresponding `LLMDataset` class
- Each batch contains individual examples that may come from different tasks
- Batches return tensors with shapes:
  ```
  {
      "input_ids": [batch_size, seq_len],
      "output_ids": [batch_size, seq_len],
      "attention_mask": [batch_size, seq_len],
      "is_test": [batch_size],
      "task_ids": [batch_size] (if include_task_id=True)
  }
  ```
- **Performance Note**: Enables efficient parallel training on GPU as all examples have uniform tensor sizes within a batch

##### With `by_task=True`:
- Uses the corresponding `TaskDataset` class
- Each batch contains complete tasks with all their examples
- Batches return dictionaries with structure:
  ```
  {
      "task_ids": [batch_size],
      "train": [  # List of training examples for each task
          {  # Task 1
              "input_ids": [num_train_examples, seq_len],
              "output_ids": [num_train_examples, seq_len],
              "attention_mask": [num_train_examples, seq_len]
          },
          # ... more tasks
      ],
      "test": [  # List of test examples for each task
          {  # Task 1
              "input_ids": [num_test_examples, seq_len],
              "output_ids": [num_test_examples, seq_len],
              "attention_mask": [num_test_examples, seq_len]
          },
          # ... more tasks
      ]
  }
  ```
- **Performance Note**: Less efficient for parallel GPU training as it requires iterating through tasks and examples separately, but preserves the complete task structure

### Batch Processing

When batching, multiple examples are combined:

1. Sequences are padded to the length of the longest sequence in the batch (not to the maximum sequence length)
2. An attention mask is created to distinguish real tokens (1) from padding tokens (0)
3. The batch-specific information is returned as a dictionary

### Tokenizer

The `ARCTokenizer` converts 2D grids to token sequences and vice versa:

```python
from arc_tokenizer import ARCTokenizer

tokenizer = ARCTokenizer()

# Grid to token sequence
grid = [[0, 1, 2], [3, 4, 5]]
input_ids = tokenizer.encode_input(grid)
output_ids = tokenizer.encode_output(grid)

# Token sequence to grid
decoded_grid = tokenizer.decode_output(output_ids)
```

## Exporting Model Predictions in JSON Format

The dataset loader provides functions for exporting model predictions in the original JSON format:

```python
from arc_utils import create_submission_file, save_predictions_to_json
from arc_tokenizer import ARCTokenizer
import torch

# Create tokenizer
tokenizer = ARCTokenizer()

# 1. For a complete submission file from a DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
create_submission_file(
    eval_dataloader=eval_dataloader,  # DataLoader with evaluation data
    model=model,                     # Trained model
    tokenizer=tokenizer,             # ARCTokenizer instance
    output_file="submission.json",   # Output file
    device=device,                   # Torch device
    max_length=100,                  # Maximum sequence length
    attempts_per_task=2,             # Number of attempts per task
    duplicate_if_fewer_attempts=True # If True, missing attempts are duplicated
)

# With beam search for multiple predictions per input
create_submission_file(
    eval_dataloader=eval_dataloader,
    model=model,
    tokenizer=tokenizer,
    output_file="beam_search_submission.json",
    device=device,
    max_length=100,
    attempts_per_task=3,             # Number of desired attempts
    duplicate_if_fewer_attempts=False, # If False, only actual predictions are stored
    num_beams=3                      # Number of beam search paths (should be ≥ attempts_per_task)
)

# 2. Or manually from existing grids
# Example predictions (already decoded grids)
predictions = {
    "task1": [[[0, 1], [2, 3]]],                 # A task with one test input
    "task2": [[[4, 5], [6, 7]], [[8, 9], [0, 1]]] # A task with two test inputs
}

# Save in submission format
save_predictions_to_json(
    predictions=predictions,
    output_file="manual_submission.json",
    attempts_per_task=2,  # Default 2 for Kaggle submissions
    duplicate_if_fewer_attempts=True  # If True, missing attempts are duplicated
)

# 3. Example with multiple attempts per test input (e.g., from beam search)
beam_search_predictions = {
    "task1": [  # A task with one test input and two attempts
        [
            [[0, 1], [2, 3]],  # Attempt 1
            [[1, 0], [3, 2]]   # Attempt 2
        ]
    ]
}

save_predictions_to_json(
    predictions=beam_search_predictions,
    output_file="beam_search_manual.json",
    attempts_per_task=2
)
```

The resulting JSON format matches the original format:

```json
{
  "task1": [
    {"attempt_1": [[0, 1], [2, 3]], "attempt_2": [[0, 1], [2, 3]]}
  ],
  "task2": [
    {"attempt_1": [[4, 5], [6, 7]], "attempt_2": [[4, 5], [6, 7]]}
  ]
}
```

When using beam search, multiple different solution attempts can be exported:

```json
{
  "task1": [
    {"attempt_1": [[0, 1], [2, 3]], "attempt_2": [[1, 0], [3, 2]]}
  ]
}
```

## Training an LLM

Here's a simple example of training a transformer model with this dataset:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from config import ARC_V1_DATA_PATH
from arc_dataset import create_arc_v1_dataloader
from model import TransformerModel  # Your model implementation

# Create dataloaders
train_dataloader = create_arc_v1_dataloader(
    data_path=ARC_V1_DATA_PATH,
    split="training",
    batch_size=32,
    shuffle=True
)

val_dataloader = create_arc_v1_dataloader(
    data_path=ARC_V1_DATA_PATH,
    split="evaluation",
    batch_size=32,
    shuffle=False
)

# Setup model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        output_ids = batch["output_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Reshape for CrossEntropyLoss
        # outputs: (batch_size, seq_len, vocab_size)
        # output_ids: (batch_size, seq_len)
        loss = criterion(
            outputs.view(-1, outputs.size(-1)),
            output_ids.view(-1)
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Training Loss: {total_loss / len(train_dataloader)}")
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            output_ids = batch["output_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                output_ids.view(-1)
            )
            
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss / len(val_dataloader)}")
```

For convenience, a simple example training script is provided in `example_training_run.py`, which demonstrates how to set up and run a training loop with the ARC datasets.

## Testing

In the `tests` folder, you'll find several test scripts that help verify the dataset loaders, tokenizer, and utilities:

- `test_v1_dataset.py`: Tests for ARC-AGI v1 dataset
- `test_v2_dataset.py`: Tests for ARC-AGI-2 v2 dataset
- `test_rearc_dataset.py`: Memory-efficient tests for RE-ARC dataset
- `test_dataloaders.py`: Tests for dataloaders with different configurations

These tests can help ensure that your setup is working correctly and provide examples of how to use different components of the library.

## License Information

- **ARC-AGI v1**: Licensed under Apache-2.0 license ([ARC-AGI Repository](https://github.com/fchollet/ARC-AGI))
- **ARC-AGI-2 v2**: Licensed under Apache-2.0 license ([ARC Prize Repository](https://github.com/arcprize/ARC-AGI-2))
- **RE-ARC**: Licensed under MIT license ([Michael Hodel's RE-ARC Repository](https://github.com/michaelhodel/re-arc)) 

## Disclaimer

This project is provided as a starting point for working with ARC datasets and is not guaranteed to be error-free. It's a side project, not a professional implementation, and is offered without warranty or guarantee of any kind. Users should verify the implementation for their specific use cases and requirements.

The code may contain bugs or inconsistencies, and performance optimizations may be needed for production use. Feel free to modify and improve the implementation to suit your needs. 