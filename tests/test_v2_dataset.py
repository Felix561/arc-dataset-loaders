"""
Test script for the ARC-AGI-2 dataset classes.
"""

import unittest
import torch
import os
import sys

# Add the main directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_dataset import ARCV2LLMDataset, ARCV2TaskDataset, create_arc_v2_dataloader, llm_collate_fn
from config import ARC_V2_DATA_PATH
from torch.utils.data import DataLoader

class TestARCV2Dataset(unittest.TestCase):
    
    def test_arcv2llm_dataset(self):
        """Tests the ARCV2LLM Dataset."""
        dataset = ARCV2LLMDataset(data_path=ARC_V2_DATA_PATH, split="training")
        
        # Check if the length of the dataset is correct
        self.assertGreater(len(dataset), 0, "Dataset should contain examples")
        
        # Check if the __getitem__ method works
        example = dataset[0]
        
        print(f"Example contains the following keys: {example.keys()}")
        print(f"Input-IDs Shape: {example['input_ids'].shape}")
        print(f"Output-Shape: {example['output_ids'].shape}")
        
        # Check if the expected keys are present
        self.assertIn("input_ids", example)
        self.assertIn("output_ids", example)
        self.assertIn("input_grid", example)
        self.assertIn("output_grid", example)
        
        # Test DataLoader with batches
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=llm_collate_fn
        )
        
        batch = next(iter(dataloader))
        print(f"Batch contains the following keys: {batch.keys()}")
        print(f"Input-IDs Shape: {batch['input_ids'].shape}")
        print(f"Output-IDs Shape: {batch['output_ids'].shape}")
        print(f"Attention-Mask Shape: {batch['attention_mask'].shape}")
    
    def test_arcv2task_dataset(self):
        """Tests the ARCV2Task Dataset."""
        dataset = ARCV2TaskDataset(data_path=ARC_V2_DATA_PATH, split="training")
        
        # Check if the length of the dataset is correct
        self.assertGreater(len(dataset), 0, "Dataset should contain tasks")
        
        # Check if the __getitem__ method works
        task = dataset[0]
        
        # Check if the expected keys are present
        self.assertIn("task_id", task)
        self.assertIn("train", task)
        self.assertIn("test", task)
        
        # Check training examples, if available
        if task["train"]:
            train_example = task["train"][0]
            self.assertIn("input_grid", train_example)
            self.assertIn("output_grid", train_example)
            self.assertIn("input_ids", train_example)
            self.assertIn("output_ids", train_example)
        
        # Check test examples, if available
        if task["test"]:
            test_example = task["test"][0]
            self.assertIn("input_grid", test_example)
            self.assertIn("input_ids", test_example)

def test_v2_task_dataset():
    """Test the ARCV2TaskDataset class."""
    # Training dataset
    train_dataset = ARCV2TaskDataset(split="training")
    print(f"Number of tasks in the training dataset: {len(train_dataset)}")
    
    # Evaluation dataset
    eval_dataset = ARCV2TaskDataset(split="evaluation")
    print(f"Number of tasks in the evaluation dataset: {len(eval_dataset)}")
    
    # Check structure of a task
    task = train_dataset[0]
    print(f"Task structure: {task.keys()}")
    print(f"Task ID: {task['task_id']}")
    print(f"Number of training examples: {len(task['train'])}")
    print(f"Number of test examples: {len(task['test'])}")
    
    # Check structure of a training example
    train_example = task['train'][0]
    print(f"Training example structure: {train_example.keys()}")
    
    # Check structure of a test example
    test_example = task['test'][0]
    print(f"Test example structure: {test_example.keys()}")

def test_v2_dataloader():
    """Test the DataLoader for ARC-AGI-2."""
    # Create DataLoader for individual examples
    train_dataloader = create_arc_v2_dataloader(split="training", batch_size=4, shuffle=False)
    batch = next(iter(train_dataloader))
    print(f"Batch structure: {batch.keys()}")
    print(f"Input-IDs Shape: {batch['input_ids'].shape}")
    print(f"Target-IDs Shape: {batch['output_ids'].shape}")
    print(f"Attention-Mask Shape: {batch['attention_mask'].shape}")
    
    # Create DataLoader for whole tasks
    task_dataloader = create_arc_v2_dataloader(split="training", batch_size=2, shuffle=False, by_task=True)
    task_batch = next(iter(task_dataloader))
    print(f"Task batch structure: {task_batch[0].keys() if isinstance(task_batch, list) else 'Dict with keys: ' + str(task_batch.keys())}")

if __name__ == "__main__":
    unittest.main() 