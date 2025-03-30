"""
ARC Dataset Loader for LLM Training.

Loads the ARC-Tasks (Abstraction and Reasoning Corpus) from JSON files
and prepares them for training an LLM model.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional, Union
from config import TokenizerConfig, ModelConfig, DatasetConfig, ARC_V1_DATA_PATH, ARC_V2_DATA_PATH, V2_TRAINING_DIR, V2_EVALUATION_DIR
from config import ARC_V1_TRAINING_CHALLENGES_FILE, ARC_V1_TRAINING_SOLUTIONS_FILE, ARC_V1_EVALUATION_CHALLENGES_FILE, ARC_V1_EVALUATION_SOLUTIONS_FILE
from config import RE_ARC_DATA_PATH, RE_ARC_TASKS_DIR, RE_ARC_TRAINING_RATIO
from arc_tokenizer import ARCTokenizer

class ARCBaseDataset:
    """Base class for ARC-Datasets with common functionality."""
    
    def __init__(self, data_path: str):
        """
        Base initialization for ARC-Datasets.
        
        Args:
            data_path: Path to the ARC dataset folder
        """
        self.data_path = data_path
        self.tokenizer = ARCTokenizer()
    
    def _load_json_file(self, file_path: str) -> Dict:
        """
        Load a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Content of the JSON file as dictionary
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading file {file_path}: {str(e)}")


class ARCV1LLMDataset(Dataset, ARCBaseDataset):
    """
    ARC-Dataset for LLM training and validation.
    
    Loads ARC-Tasks from the training or evaluation set and prepares
    them for training an LLM model, where each task is converted into
    training examples (input/output pairs).
    
    Each query returns a training example:
    - Input: Tokenized input grid for the LLM
    - Target: Tokenized output grid for the LLM
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = DatasetConfig.SPLIT,
        include_task_id: bool = DatasetConfig.INCLUDE_TASK_ID,
        max_seq_length: int = TokenizerConfig.MAX_SEQ_LENGTH
    ):
        """
        Initialize the ARC-Dataset for LLM training.
        
        Args:
            data_path: Path to the ARC dataset folder
            split: Either "training" or "evaluation"
            include_task_id: Whether to include the task ID in each example
            max_seq_length: Maximum length of tokenized sequences
        """
        ARCBaseDataset.__init__(self, data_path)
        
        self.split = split.lower()
        if self.split not in ["training", "evaluation"]:
            raise ValueError(f"Split must be 'training' or 'evaluation', not '{split}'")
            
        self.include_task_id = include_task_id
        self.max_seq_length = max_seq_length
        
        # Load Challenges and Solutions
        self._load_data()
        
        # Create list of all training examples
        self._prepare_examples()
    
    def _load_data(self):
        """Load the corresponding JSON files based on the split."""
        if self.split == "training":
            challenges_file = os.path.join(self.data_path, ARC_V1_TRAINING_CHALLENGES_FILE)
            solutions_file = os.path.join(self.data_path, ARC_V1_TRAINING_SOLUTIONS_FILE)
        else:  # evaluation
            challenges_file = os.path.join(self.data_path, ARC_V1_EVALUATION_CHALLENGES_FILE)
            solutions_file = os.path.join(self.data_path, ARC_V1_EVALUATION_SOLUTIONS_FILE)
        
        self.challenges = self._load_json_file(challenges_file)
        self.solutions = self._load_json_file(solutions_file)
        
        # List of all task IDs
        self.task_ids = sorted(list(self.challenges.keys()))
    
    def _prepare_examples(self):
        """
        Prepare all training examples.
        
        Each example consists of an input (grid) and target (grid).
        """
        self.examples = []
        
        for task_id in self.task_ids:
            task = self.challenges[task_id]
            
            # Training examples for this task
            for train_example in task.get("train", []):
                self.examples.append({
                    "task_id": task_id,
                    "input_grid": train_example["input"],
                    "output_grid": train_example["output"],
                    "is_test": False
                })
            
            # Test examples for this task (for validation)
            for test_idx, test_example in enumerate(task.get("test", [])):
                if task_id in self.solutions:
                    # Make sure there is a solution for this test example
                    if test_idx < len(self.solutions[task_id]):
                        self.examples.append({
                            "task_id": task_id,
                            "input_grid": test_example["input"],
                            "output_grid": self.solutions[task_id][test_idx],
                            "is_test": True
                        })
    
    def __len__(self) -> int:
        """Returns the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns an example from the dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary with tokenized input and output sequences
        """
        example = self.examples[idx]
        
        # Tokenize input and output
        input_ids = self.tokenizer.encode_input(example["input_grid"])
        output_ids = self.tokenizer.encode_output(example["output_grid"])
        
        # Truncate sequences that are too long
        input_ids = input_ids[:self.max_seq_length]
        output_ids = output_ids[:self.max_seq_length]
        
        # Create the result dictionary
        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "output_ids": torch.tensor(output_ids, dtype=torch.long),
            "input_grid": example["input_grid"],
            "output_grid": example["output_grid"],
            "is_test": example["is_test"]
        }
        
        if self.include_task_id:
            result["task_id"] = example["task_id"]
        
        return result


class ARCV1TaskDataset(Dataset, ARCBaseDataset):
    """
    ARC-Dataset for Tasks.
    
    Loads complete ARC-Tasks (with all training and test examples)
    and returns them as individual tasks. This is useful for
    evaluation and for few-shot learning approaches.
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = DatasetConfig.SPLIT,
        max_seq_length: int = TokenizerConfig.MAX_SEQ_LENGTH
    ):
        """
        Initialize the ARC-Task-Dataset.
        
        Args:
            data_path: Path to the ARC dataset folder
            split: Either "training" or "evaluation"
            max_seq_length: Maximum length of tokenized sequences
        """
        ARCBaseDataset.__init__(self, data_path)
        
        self.split = split.lower()
        if self.split not in ["training", "evaluation"]:
            raise ValueError(f"Split must be 'training' or 'evaluation', not '{split}'")
            
        self.max_seq_length = max_seq_length
        
        # Load Challenges and Solutions
        self._load_data()
    
    def _load_data(self):
        """Load the corresponding JSON files based on the split."""
        if self.split == "training":
            challenges_file = os.path.join(self.data_path, ARC_V1_TRAINING_CHALLENGES_FILE)
            solutions_file = os.path.join(self.data_path, ARC_V1_TRAINING_SOLUTIONS_FILE)
        else:  # evaluation
            challenges_file = os.path.join(self.data_path, ARC_V1_EVALUATION_CHALLENGES_FILE)
            solutions_file = os.path.join(self.data_path, ARC_V1_EVALUATION_SOLUTIONS_FILE)
        
        self.challenges = self._load_json_file(challenges_file)
        self.solutions = self._load_json_file(solutions_file)
        
        # List of all task IDs
        self.task_ids = sorted(list(self.challenges.keys()))
    
    def __len__(self) -> int:
        """Returns the number of tasks in the dataset."""
        return len(self.task_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a complete task from the dataset.
        
        Args:
            idx: Index of the task
            
        Returns:
            Dictionary with all information about the task
        """
        task_id = self.task_ids[idx]
        task = self.challenges[task_id]
        
        # Training examples
        train_examples = []
        for example in task.get("train", []):
            # Tokenize with maximum length
            input_ids = self.tokenizer.encode_input(example["input"])[:self.max_seq_length]
            output_ids = self.tokenizer.encode_output(example["output"])[:self.max_seq_length]
            
            train_examples.append({
                "input_grid": example["input"],
                "output_grid": example["output"],
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "output_ids": torch.tensor(output_ids, dtype=torch.long)
            })
        
        # Test examples
        test_examples = []
        for test_idx, test_example in enumerate(task.get("test", [])):
            # Input grid
            input_grid = test_example["input"]
            
            # Tokenize
            input_ids = self.tokenizer.encode_input(input_grid)[:self.max_seq_length]
            
            test_example_dict = {
                "input_grid": input_grid,
                "input_ids": torch.tensor(input_ids, dtype=torch.long)
            }
            
            # Add output only if available
            if "output" in test_example:
                output_grid = test_example["output"]
                output_ids = self.tokenizer.encode_output(output_grid)[:self.max_seq_length]
                test_example_dict["output_grid"] = output_grid
                test_example_dict["output_ids"] = torch.tensor(output_ids, dtype=torch.long)
            
            test_examples.append(test_example_dict)
        
        return {
            "task_id": task_id,
            "train": train_examples,
            "test": test_examples
        }
    
    def get_task_by_id(self, task_id: str) -> Dict[str, Any]:
        """
        Returns a task by its ID.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dictionary with all information about the task
        """
        if task_id not in self.challenges:
            raise ValueError(f"Task ID {task_id} not found in the {self.split} set")
        
        task_idx = self.task_ids.index(task_id)
        return self.__getitem__(task_idx)


def llm_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collation function for LLM batches.
    
    Pads sequences to the same length for batch processing.
    
    Args:
        batch: List of examples from the dataset
    
    Returns:
        Dictionary with padded tensors for batch processing
    """
    # Find the maximum length of sequences in the batch
    max_input_len = max(len(x["input_ids"]) for x in batch)
    max_output_len = max(len(x["output_ids"]) for x in batch)
    
    # Initialize lists for the padded sequences
    input_ids = []
    output_ids = []
    attention_mask = []
    is_test = []
    
    # Optional lists, if available
    task_ids = []
    
    for sample in batch:
        # Convert to lists if they are tensors
        sample_input_ids = sample["input_ids"].tolist()
        sample_output_ids = sample["output_ids"].tolist()
        
        # Create padding for input IDs and attention mask
        padded_input_ids = sample_input_ids + [TokenizerConfig.PAD_ID] * (max_input_len - len(sample_input_ids))
        sample_attention_mask = [1] * len(sample_input_ids) + [0] * (max_input_len - len(sample_input_ids))
        
        # Create padding for output IDs
        padded_output_ids = sample_output_ids + [TokenizerConfig.PAD_ID] * (max_output_len - len(sample_output_ids))
        
        # Add padded sequences to the lists
        input_ids.append(padded_input_ids)
        output_ids.append(padded_output_ids)
        attention_mask.append(sample_attention_mask)
        is_test.append(sample["is_test"])
        
        # Add optional task ID if available
        if "task_id" in sample:
            task_ids.append(sample["task_id"])
    
    # Convert lists to tensors
    result = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "output_ids": torch.tensor(output_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "is_test": torch.tensor(is_test, dtype=torch.bool)
    }
    
    if task_ids:
        result["task_ids"] = task_ids
    
    return result


def task_collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collate function for Task-Datasets that return complete tasks.
    This function maintains the original structure of the tasks.
    
    Args:
        batch: List of task dictionaries
        
    Returns:
        List of task dictionaries without changes
    """
    # For task datasets we simply return the list of tasks
    return batch


class ARCV2LLMDataset(Dataset, ARCBaseDataset):
    """
    ARC-AGI-2 Dataset for LLM training and validation.
    
    Loads ARC-AGI-2 Tasks from the training or evaluation set and prepares
    them for training an LLM model, where each task is converted into
    training examples (input/output pairs).
    
    Each query returns a training example:
    - Input: Tokenized input grid for the LLM
    - Target: Tokenized output grid for the LLM
    """
    
    def __init__(
        self,
        data_path: str = DatasetConfig.ARCV2.DATA_PATH,
        split: str = DatasetConfig.SPLIT,
        include_task_id: bool = DatasetConfig.INCLUDE_TASK_ID,
        max_seq_length: int = TokenizerConfig.MAX_SEQ_LENGTH
    ):
        """
        Initialize the ARC-AGI-2 Dataset for LLM training.
        
        Args:
            data_path: Path to the ARC-AGI-2 dataset folder
            split: Either "training" or "evaluation"
            include_task_id: Whether to include the task ID in each example
            max_seq_length: Maximum length of tokenized sequences
        """
        ARCBaseDataset.__init__(self, data_path)
        
        self.split = split.lower()
        if self.split not in ["training", "evaluation"]:
            raise ValueError(f"Split must be 'training' or 'evaluation', not '{split}'")
            
        self.include_task_id = include_task_id
        self.max_seq_length = max_seq_length
        
        # Load Tasks based on the split
        self._load_data()
        
        # Create list of all training examples
        self._prepare_examples()
    
    def _load_data(self):
        """Load all JSON files from the corresponding directory."""
        if self.split == "training":
            task_dir = os.path.join(self.data_path, V2_TRAINING_DIR)
        else:  # evaluation
            task_dir = os.path.join(self.data_path, V2_EVALUATION_DIR)
        
        self.tasks = {}
        
        # Go through all JSON files in the directory
        for filename in os.listdir(task_dir):
            if filename.endswith('.json'):
                task_id = os.path.splitext(filename)[0]  # Remove .json, use the rest as task_id
                file_path = os.path.join(task_dir, filename)
                try:
                    self.tasks[task_id] = self._load_json_file(file_path)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
        
        # List of all task IDs
        self.task_ids = sorted(list(self.tasks.keys()))
    
    def _prepare_examples(self):
        """
        Prepare all training examples.
        
        Each example consists of an input (grid) and target (grid).
        """
        self.examples = []
        
        for task_id in self.task_ids:
            task = self.tasks[task_id]
            
            # Training examples for this task
            for train_example in task.get("train", []):
                self.examples.append({
                    "task_id": task_id,
                    "input_grid": train_example["input"],
                    "output_grid": train_example["output"],
                    "is_test": False
                })
            
            # Test examples for this task (only use for validation if the output grid is available)
            for test_example in task.get("test", []):
                if "output" in test_example:  # Only use if the output grid is known
                    self.examples.append({
                        "task_id": task_id,
                        "input_grid": test_example["input"],
                        "output_grid": test_example["output"],
                        "is_test": True
                    })
    
    def __len__(self) -> int:
        """Returns the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns an example from the dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary with tokenized input and output sequences
        """
        example = self.examples[idx]
        
        # Tokenize input and output
        input_ids = self.tokenizer.encode_input(example["input_grid"])
        output_ids = self.tokenizer.encode_output(example["output_grid"])
        
        # Truncate sequences that are too long
        input_ids = input_ids[:self.max_seq_length]
        output_ids = output_ids[:self.max_seq_length]
        
        # Create the result dictionary
        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "output_ids": torch.tensor(output_ids, dtype=torch.long),
            "input_grid": example["input_grid"],
            "output_grid": example["output_grid"],
            "is_test": example["is_test"]
        }
        
        if self.include_task_id:
            result["task_id"] = example["task_id"]
        
        return result


class ARCV2TaskDataset(Dataset, ARCBaseDataset):
    """
    ARC-AGI-2 Dataset for Tasks.
    
    Loads complete ARC-AGI-2 Tasks (with all training and test examples)
    and returns them as individual tasks. This is useful for
    evaluation and for few-shot learning approaches.
    """
    
    def __init__(
        self,
        data_path: str = DatasetConfig.ARCV2.DATA_PATH,
        split: str = DatasetConfig.SPLIT,
        max_seq_length: int = TokenizerConfig.MAX_SEQ_LENGTH
    ):
        """
        Initialize the ARC-AGI-2 Task-Dataset.
        
        Args:
            data_path: Path to the ARC-AGI-2 dataset folder
            split: Either "training" or "evaluation"
            max_seq_length: Maximum length of tokenized sequences
        """
        ARCBaseDataset.__init__(self, data_path)
        
        self.split = split.lower()
        if self.split not in ["training", "evaluation"]:
            raise ValueError(f"Split must be 'training' or 'evaluation', not '{split}'")
            
        self.max_seq_length = max_seq_length
        
        # Load Tasks based on the split
        self._load_data()
    
    def _load_data(self):
        """Load all JSON files from the corresponding directory."""
        if self.split == "training":
            task_dir = os.path.join(self.data_path, V2_TRAINING_DIR)
        else:  # evaluation
            task_dir = os.path.join(self.data_path, V2_EVALUATION_DIR)
        
        self.tasks = {}
        
        # Go through all JSON files in the directory
        for filename in os.listdir(task_dir):
            if filename.endswith('.json'):
                task_id = os.path.splitext(filename)[0]  # Remove .json, use the rest as task_id
                file_path = os.path.join(task_dir, filename)
                try:
                    self.tasks[task_id] = self._load_json_file(file_path)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
        
        # List of all task IDs
        self.task_ids = sorted(list(self.tasks.keys()))
    
    def __len__(self) -> int:
        """Returns the number of tasks in the dataset."""
        return len(self.task_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a complete task from the dataset.
        
        Args:
            idx: Index of the task
            
        Returns:
            Dictionary with the complete task, including tokenized sequences
        """
        task_id = self.task_ids[idx]
        task_data = self.tasks[task_id]
        
        # Tokenize training examples
        train_examples = []
        for example in task_data.get("train", []):
            # Input/Output grids
            input_grid = example["input"]
            output_grid = example["output"]
            
            # Tokenize
            input_ids = self.tokenizer.encode_input(input_grid)[:self.max_seq_length]
            output_ids = self.tokenizer.encode_output(output_grid)[:self.max_seq_length]
            
            train_examples.append({
                "input_grid": input_grid,
                "output_grid": output_grid,
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "output_ids": torch.tensor(output_ids, dtype=torch.long)
            })
        
        # Tokenize test examples
        test_examples = []
        for example in task_data.get("test", []):
            # Input grid
            input_grid = example["input"]
            
            # Tokenize
            input_ids = self.tokenizer.encode_input(input_grid)[:self.max_seq_length]
            
            test_example = {
                "input_grid": input_grid,
                "input_ids": torch.tensor(input_ids, dtype=torch.long)
            }
            
            # Add output only if available
            if "output" in example:
                output_grid = example["output"]
                output_ids = self.tokenizer.encode_output(output_grid)[:self.max_seq_length]
                test_example["output_grid"] = output_grid
                test_example["output_ids"] = torch.tensor(output_ids, dtype=torch.long)
            
            test_examples.append(test_example)
        
        return {
            "task_id": task_id,
            "train": train_examples,
            "test": test_examples
        }
    
    def get_task_by_id(self, task_id: str) -> Dict[str, Any]:
        """
        Returns a task by its ID.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dictionary with the complete task, including tokenized sequences
        
        Raises:
            ValueError: If the task is not found
        """
        if task_id not in self.task_ids:
            raise ValueError(f"Task ID {task_id} not found")
        
        # Find the index of the task and use __getitem__
        idx = self.task_ids.index(task_id)
        return self.__getitem__(idx)


def create_arc_v1_dataloader(
    data_path: str,
    split: str = DatasetConfig.SPLIT,
    batch_size: int = ModelConfig.BATCH_SIZE,
    shuffle: bool = DatasetConfig.SHUFFLE_DATA,
    num_workers: int = ModelConfig.NUM_WORKERS,
    by_task: bool = DatasetConfig.BY_TASK,
    include_task_id: bool = DatasetConfig.INCLUDE_TASK_ID
) -> DataLoader:
    """
    Creates a DataLoader for the ARC v1 dataset.
    
    Args:
        data_path: Path to the ARC dataset folder
        split: Either "training" or "evaluation"
        batch_size: Size of the batches
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for loading the data
        by_task: Whether to load complete tasks instead of individual examples
        include_task_id: Whether to include the task ID in each example
    
    Returns:
        PyTorch DataLoader for the specified dataset
    """
    if by_task:
        dataset = ARCV1TaskDataset(
            data_path=data_path, 
            split=split
        )
        collate_fn = task_collate_fn
    else:
        dataset = ARCV1LLMDataset(
            data_path=data_path, 
            split=split,
            include_task_id=include_task_id
        )
        collate_fn = llm_collate_fn
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


def create_arc_v2_dataloader(
    data_path: str = DatasetConfig.ARCV2.DATA_PATH,
    split: str = DatasetConfig.SPLIT,
    batch_size: int = ModelConfig.BATCH_SIZE,
    shuffle: bool = DatasetConfig.SHUFFLE_DATA,
    num_workers: int = ModelConfig.NUM_WORKERS,
    by_task: bool = DatasetConfig.BY_TASK,
    include_task_id: bool = DatasetConfig.INCLUDE_TASK_ID
) -> DataLoader:
    """
    Creates a DataLoader for the ARC-AGI-2 dataset.
    
    Args:
        data_path: Path to the ARC-AGI-2 dataset folder
        split: Either "training" or "evaluation"
        batch_size: Size of the batches
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for loading the data
        by_task: Whether to load complete tasks instead of individual examples
        include_task_id: Whether to include the task ID in each example
    
    Returns:
        PyTorch DataLoader for the specified dataset
    """
    if by_task:
        dataset = ARCV2TaskDataset(
            data_path=data_path, 
            split=split
        )
        collate_fn = task_collate_fn  # Use the special Collate function for tasks
    else:
        dataset = ARCV2LLMDataset(
            data_path=data_path, 
            split=split,
            include_task_id=include_task_id
        )
        collate_fn = llm_collate_fn  # Use the normal Collate function for LLM examples
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


class REARCLLMDataset(Dataset, ARCBaseDataset):
    """
    RE-ARC Dataset for LLM training and validation.
    
    Loads RE-ARC Tasks from the tasks directory and prepares
    them for training an LLM model, where each task is converted into
    training examples (input/output pairs).
    
    Each query returns a training example:
    - Input: Tokenized input grid for the LLM
    - Target: Tokenized output grid for the LLM
    """
    
    def __init__(
        self,
        data_path: str = DatasetConfig.REARC.DATA_PATH,
        split: str = DatasetConfig.SPLIT,
        include_task_id: bool = DatasetConfig.INCLUDE_TASK_ID,
        max_seq_length: int = TokenizerConfig.MAX_SEQ_LENGTH,
        task_ids: Optional[List[str]] = None,
        seed: int = DatasetConfig.REARC.SEED
    ):
        """
        Initialize the RE-ARC Dataset for LLM training.
        
        Args:
            data_path: Path to the RE-ARC dataset folder
            split: Either "training" or "evaluation"
            include_task_id: Whether to include the task ID in each example
            max_seq_length: Maximum length of tokenized sequences
            task_ids: Optional list of task IDs to be used
            seed: Random Seed for data splitting
        """
        ARCBaseDataset.__init__(self, data_path)
        
        self.split = split.lower()
        if self.split not in ["training", "evaluation"]:
            raise ValueError(f"Split must be 'training' or 'evaluation', not '{split}'")
            
        self.include_task_id = include_task_id
        self.max_seq_length = max_seq_length
        self.task_ids = task_ids
        self.seed = seed
        
        # Load Tasks based on the split
        self._load_data()
        
        # Create list of all training examples
        self._prepare_examples()
    
    def _load_data(self):
        """Load the task files and split them into training and evaluation."""
        tasks_dir = os.path.join(self.data_path, RE_ARC_TASKS_DIR)
        
        # Load all task files
        all_task_files = [f for f in os.listdir(tasks_dir) if f.endswith('.json')]
        
        # If no specific task IDs are given, split all tasks evenly
        if self.task_ids is None:
            # Set the seed for reproducibility
            np.random.seed(self.seed)
            
            # Shuffle the task files
            np.random.shuffle(all_task_files)
            
            # Split into training and evaluation
            split_idx = int(len(all_task_files) * RE_ARC_TRAINING_RATIO)
            
            if self.split == "training":
                task_files = all_task_files[:split_idx]
            else:  # evaluation
                task_files = all_task_files[split_idx:]
        else:
            # Filter by the specified task IDs
            task_files = [f for f in all_task_files if f.split('.')[0] in self.task_ids]
        
        self.tasks = {}
        
        # Load the selected task files
        for filename in task_files:
            task_id = os.path.splitext(filename)[0]
            file_path = os.path.join(tasks_dir, filename)
            try:
                task_data = self._load_json_file(file_path)
                self.tasks[task_id] = task_data
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        # List of all loaded task IDs
        self.loaded_task_ids = sorted(list(self.tasks.keys()))
    
    def _prepare_examples(self):
        """
        Prepare all training examples.
        
        Each example consists of an input (grid) and target (grid).
        """
        self.examples = []
        
        for task_id in self.loaded_task_ids:
            task_examples = self.tasks[task_id]
            
            # Each file contains an array of examples
            for example in task_examples:
                if "input" in example and "output" in example:
                    self.examples.append({
                        "task_id": task_id,
                        "input_grid": example["input"],
                        "output_grid": example["output"],
                        "is_test": False  # All examples are synthetically generated
                    })
    
    def __len__(self) -> int:
        """Returns the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns an example from the dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary with tokenized input and output sequences
        """
        example = self.examples[idx]
        
        # Tokenize input and output
        input_ids = self.tokenizer.encode_input(example["input_grid"])
        output_ids = self.tokenizer.encode_output(example["output_grid"])
        
        # Truncate sequences that are too long
        input_ids = input_ids[:self.max_seq_length]
        output_ids = output_ids[:self.max_seq_length]
        
        # Create the result dictionary
        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "output_ids": torch.tensor(output_ids, dtype=torch.long),
            "input_grid": example["input_grid"],
            "output_grid": example["output_grid"],
            "is_test": example["is_test"]
        }
        
        if self.include_task_id:
            result["task_id"] = example["task_id"]
        
        return result


class REARCTaskDataset(Dataset, ARCBaseDataset):
    """
    RE-ARC Dataset for Tasks.
    
    Loads complete RE-ARC Tasks (with all generated examples)
    and returns them as individual tasks. This is useful for
    evaluation and for few-shot learning approaches.
    """
    
    def __init__(
        self,
        data_path: str = DatasetConfig.REARC.DATA_PATH,
        split: str = DatasetConfig.SPLIT,
        max_seq_length: int = TokenizerConfig.MAX_SEQ_LENGTH,
        task_ids: Optional[List[str]] = None,
        examples_per_task: int = DatasetConfig.REARC.EXAMPLES_PER_TASK,
        seed: int = DatasetConfig.REARC.SEED
    ):
        """
        Initialize the RE-ARC Task-Dataset.
        
        Args:
            data_path: Path to the RE-ARC dataset folder
            split: Either "training" or "evaluation"
            max_seq_length: Maximum length of tokenized sequences
            task_ids: Optional list of task IDs to be used
            examples_per_task: Number of examples to be loaded for each task
            seed: Random Seed for data splitting
        """
        ARCBaseDataset.__init__(self, data_path)
        
        self.split = split.lower()
        if self.split not in ["training", "evaluation"]:
            raise ValueError(f"Split must be 'training' or 'evaluation', not '{split}'")
            
        self.max_seq_length = max_seq_length
        self.task_ids = task_ids
        self.examples_per_task = examples_per_task
        self.seed = seed
        
        # Load Tasks based on the split
        self._load_data()
    
    def _load_data(self):
        """Load the task files and split them into training and evaluation."""
        tasks_dir = os.path.join(self.data_path, RE_ARC_TASKS_DIR)
        
        # Load all task files
        all_task_files = [f for f in os.listdir(tasks_dir) if f.endswith('.json')]
        
        # If no specific task IDs are given, split all tasks evenly
        if self.task_ids is None:
            # Set the seed for reproducibility
            np.random.seed(self.seed)
            
            # Shuffle the task files
            np.random.shuffle(all_task_files)
            
            # Split into training and evaluation
            split_idx = int(len(all_task_files) * RE_ARC_TRAINING_RATIO)
            
            if self.split == "training":
                task_files = all_task_files[:split_idx]
            else:  # evaluation
                task_files = all_task_files[split_idx:]
        else:
            # Filter by the specified task IDs
            task_files = [f for f in all_task_files if f.split('.')[0] in self.task_ids]
        
        self.tasks = {}
        
        # Load the selected task files
        for filename in task_files:
            task_id = os.path.splitext(filename)[0]
            file_path = os.path.join(tasks_dir, filename)
            try:
                # Load the task data
                task_data = self._load_json_file(file_path)
                
                # Set the seed for reproducibility
                np.random.seed(self.seed + hash(task_id) % 10000)
                
                # Select a subset of examples (if more available)
                if len(task_data) > self.examples_per_task:
                    selected_indices = np.random.choice(
                        len(task_data),
                        size=self.examples_per_task,
                        replace=False
                    )
                    task_data = [task_data[i] for i in selected_indices]
                
                self.tasks[task_id] = task_data
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        # List of all loaded task IDs
        self.loaded_task_ids = sorted(list(self.tasks.keys()))
    
    def __len__(self) -> int:
        """Returns the number of tasks in the dataset."""
        return len(self.loaded_task_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a complete task from the dataset.
        
        Args:
            idx: Index of the task
            
        Returns:
            Dictionary with all information about the task
        """
        task_id = self.loaded_task_ids[idx]
        task_examples = self.tasks[task_id]
        
        # RE-ARC has no distinction between train and test
        # We split the examples evenly
        # By default: 80% train, 20% test
        np.random.seed(self.seed + hash(task_id) % 10000)
        indices = np.random.permutation(len(task_examples))
        
        split_idx = int(len(task_examples) * 0.8)  # 80% for training
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        # Training examples
        train_examples = []
        for i in train_indices:
            example = task_examples[i]
            
            # Tokenize input and output
            input_ids = self.tokenizer.encode_input(example["input"])[:self.max_seq_length]
            output_ids = self.tokenizer.encode_output(example["output"])[:self.max_seq_length]
            
            train_examples.append({
                "input_grid": example["input"],
                "output_grid": example["output"],
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "output_ids": torch.tensor(output_ids, dtype=torch.long)
            })
        
        # Test examples
        test_examples = []
        for i in test_indices:
            example = task_examples[i]
            
            # Tokenize input and output
            input_ids = self.tokenizer.encode_input(example["input"])[:self.max_seq_length]
            output_ids = self.tokenizer.encode_output(example["output"])[:self.max_seq_length]
            
            test_examples.append({
                "input_grid": example["input"],
                "output_grid": example["output"],
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "output_ids": torch.tensor(output_ids, dtype=torch.long)
            })
        
        return {
            "task_id": task_id,
            "train": train_examples,
            "test": test_examples
        }
    
    def get_task_by_id(self, task_id: str) -> Dict[str, Any]:
        """
        Returns a task by its ID.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dictionary with all information about the task
        """
        if task_id not in self.loaded_task_ids:
            raise ValueError(f"Task ID {task_id} not found in the {self.split} set")
        
        task_idx = self.loaded_task_ids.index(task_id)
        return self.__getitem__(task_idx)


def create_re_arc_dataloader(
    data_path: str = DatasetConfig.REARC.DATA_PATH,
    split: str = DatasetConfig.SPLIT,
    batch_size: int = ModelConfig.BATCH_SIZE,
    shuffle: bool = DatasetConfig.SHUFFLE_DATA,
    num_workers: int = ModelConfig.NUM_WORKERS,
    by_task: bool = DatasetConfig.BY_TASK,
    task_ids: Optional[List[str]] = None,
    examples_per_task: int = DatasetConfig.REARC.EXAMPLES_PER_TASK,
    seed: int = DatasetConfig.REARC.SEED,
    include_task_id: bool = DatasetConfig.INCLUDE_TASK_ID
) -> DataLoader:
    """
    Creates a DataLoader for the RE-ARC dataset.
    
    Args:
        data_path: Path to the RE-ARC dataset folder
        split: Either "training" or "evaluation"
        batch_size: Size of the batches
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for loading the data
        by_task: Whether to load complete tasks instead of individual examples
        task_ids: Optional list of task IDs to be used
        examples_per_task: Number of examples to be loaded for each task (only for by_task=True)
        seed: Random Seed for data splitting
        include_task_id: Whether to include the task ID in each example
    
    Returns:
        PyTorch DataLoader for the specified dataset
    """
    if by_task:
        dataset = REARCTaskDataset(
            data_path=data_path, 
            split=split, 
            task_ids=task_ids,
            examples_per_task=examples_per_task,
            seed=seed
        )
        collate_fn = task_collate_fn  # Use the special Collate function for tasks
    else:
        dataset = REARCLLMDataset(
            data_path=data_path, 
            split=split,
            task_ids=task_ids,
            seed=seed,
            include_task_id=include_task_id
        )
        collate_fn = llm_collate_fn  # Use the normal Collate function for LLM examples
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


# Example usage
if __name__ == "__main__":
    from config import ARC_V1_DATA_PATH, RE_ARC_DATA_PATH
    
    # Example for ARC v1
    # Create dataset for training (individual examples)
    train_dataset = ARCV1LLMDataset(ARC_V1_DATA_PATH, split="training")
    print(f"Number of training examples (ARC v1): {len(train_dataset)}")
    
    # Get an example
    example = train_dataset[0]
    print(f"Example Task-ID: {example.get('task_id', 'N/A')}")
    print(f"Input IDs Length: {len(example['input_ids'])}")
    print(f"Target IDs Length: {len(example['output_ids'])}")
    print(f"Is Test Example: {example['is_test']}")
    
    # Create dataset for complete tasks
    task_dataset = ARCV1TaskDataset(ARC_V1_DATA_PATH, split="training")
    print(f"Number of tasks (ARC v1): {len(task_dataset)}")
    
    # Get a task
    task = task_dataset[0]
    print(f"Task-ID: {task['task_id']}")
    print(f"Number of training examples: {len(task['train'])}")
    print(f"Number of test examples: {len(task['test'])}")
    
    # Create DataLoader for training
    train_dataloader = create_arc_v1_dataloader(
        data_path=ARC_V1_DATA_PATH,
        split="training",
        batch_size=4,
        shuffle=True
    )
    
    # Get a batch
    batch = next(iter(train_dataloader))
    print(f"Batch Input-IDs Shape: {batch['input_ids'].shape}")
    print(f"Batch Target-IDs Shape: {batch['output_ids'].shape}")
    print(f"Batch Attention-Mask Shape: {batch['attention_mask'].shape}") 
    
    # Example for RE-ARC
    print("\n--- RE-ARC Dataset Example ---")
    
    # Create RE-ARC dataset for training (individual examples)
    re_arc_train_dataset = REARCLLMDataset(RE_ARC_DATA_PATH, split="training")
    print(f"Number of training examples (RE-ARC): {len(re_arc_train_dataset)}")
    
    # Create RE-ARC dataset for tasks
    re_arc_task_dataset = REARCTaskDataset(RE_ARC_DATA_PATH, split="training", examples_per_task=5)
    print(f"Number of tasks (RE-ARC): {len(re_arc_task_dataset)}")
    
    if len(re_arc_task_dataset) > 0:
        # Get a task
        re_arc_task = re_arc_task_dataset[0]
        print(f"RE-ARC Task-ID: {re_arc_task['task_id']}")
        print(f"Number of training examples: {len(re_arc_task['train'])}")
        print(f"Number of test examples: {len(re_arc_task['test'])}")

    # Create DataLoader for RE-ARC training (examples)
    re_arc_train_dataloader = create_re_arc_dataloader(
        split="training",
        batch_size=8,
        shuffle=True,
        by_task=False
    )
    
    # Create DataLoader for RE-ARC training (tasks)
    re_arc_task_dataloader = create_re_arc_dataloader(
        split="training",
        batch_size=4,
        shuffle=True,
        by_task=True,
        examples_per_task=10
    ) 