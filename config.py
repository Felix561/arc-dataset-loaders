"""
Configuration file for ARC dataset and tokenizer.
"""

# Path to ARC dataset (v1)
ARC_V1_DATA_PATH = "arc_v1_data"

# Path to ARC-AGI-2 dataset (v2)
ARC_V2_DATA_PATH = "arc_v2_data"

# Path to RE-ARC dataset
RE_ARC_DATA_PATH = "re_arc_data"

# Filenames for ARC v1
ARC_V1_TRAINING_CHALLENGES_FILE = "arc-agi_training_challenges.json"
ARC_V1_TRAINING_SOLUTIONS_FILE = "arc-agi_training_solutions.json"
ARC_V1_EVALUATION_CHALLENGES_FILE = "arc-agi_evaluation_challenges.json"
ARC_V1_EVALUATION_SOLUTIONS_FILE = "arc-agi_evaluation_solutions.json"

# Directory names for ARC-AGI-2 (v2)
V2_TRAINING_DIR = "training"
V2_EVALUATION_DIR = "evaluation"

# RE-ARC configuration
RE_ARC_TASKS_DIR = "tasks"
RE_ARC_METADATA_FILE = "metadata.json"
RE_ARC_TRAINING_RATIO = 0.8  # 80% of tasks for training, 20% for validation

# Tokenizer configuration
class TokenizerConfig:
    # Special tokens
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    
    # Color tokens (0-9)
    COLOR_TOKENS = [str(i) for i in range(10)]
    
    # Newline token
    NEWLINE_TOKEN = "\n"
    
    # Complete input vocabulary
    INPUT_VOCAB = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, NEWLINE_TOKEN] + COLOR_TOKENS
    
    # Complete output vocabulary (no BOS token)
    OUTPUT_VOCAB = [PAD_TOKEN, EOS_TOKEN, NEWLINE_TOKEN] + COLOR_TOKENS
    
    # Token IDs
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    NEWLINE_ID = 3
    
    # Color token IDs start at 4
    COLOR_ID_START = 4
    
    # Maximum length for input/output sequences
    MAX_SEQ_LENGTH = 1024  # Can be adjusted according to requirements

# Model configuration
class ModelConfig:
    # General configuration for training
    BATCH_SIZE = 32         # Batch size for training
    NUM_WORKERS = 8         # Number of workers for DataLoader
    
    # Training configuration
    LEARNING_RATE = 1e-4    # Learning rate for optimizer
    WEIGHT_DECAY = 0.01     # Weight decay for regularization
    NUM_EPOCHS = 10         # Number of training epochs
    
    # Maximum grid size
    MAX_GRID_SIZE = 30

# Dataset-specific configurations
class DatasetConfig:
    # General configuration for all datasets
    SPLIT = "training"      # "training" or "evaluation"
    SHUFFLE_DATA = True     # Whether to shuffle the data
    BY_TASK = False         # Whether to load complete tasks (True) or individual examples (False)
    INCLUDE_TASK_ID = True  # Whether to include the task ID in each example
    
    # ARC-V1 configuration
    class ARCV1:
        DATA_PATH = ARC_V1_DATA_PATH
    
    # ARC-V2 configuration
    class ARCV2:
        DATA_PATH = ARC_V2_DATA_PATH
    
    # RE-ARC configuration
    class REARC:
        DATA_PATH = RE_ARC_DATA_PATH
        EXAMPLES_PER_TASK = 5  # Number of examples per task for REARCTaskDataset
        SEED = 42             # Random seed for data splitting 