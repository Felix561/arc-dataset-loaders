"""
Test script for the RE-ARC dataset classes.
Memory-efficient implementation that loads only a subset of tasks.
"""

import unittest
import torch
import os
import sys
import numpy as np

# Add the main directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from arc_dataset import REARCLLMDataset, REARCTaskDataset, create_re_arc_dataloader, llm_collate_fn
from config import DatasetConfig
from torch.utils.data import DataLoader

# Define explicit paths relative to the parent directory
RE_ARC_DATA_PATH = os.path.join(parent_dir, 're_arc_data')

# Number of tasks to load for testing to limit memory usage
MAX_TEST_TASKS = 3

def get_limited_task_ids(data_path, max_tasks=MAX_TEST_TASKS):
    """Get a limited number of task IDs to reduce memory usage."""
    tasks_dir = os.path.join(data_path, "tasks")
    if not os.path.exists(tasks_dir) or not os.path.isdir(tasks_dir):
        raise FileNotFoundError(f"RE-ARC tasks directory not found at {tasks_dir}")
    
    all_task_files = [f for f in os.listdir(tasks_dir) if f.endswith('.json')]
    # Set seed for reproducibility
    np.random.seed(42)
    # Randomly select a small number of tasks
    selected_files = np.random.choice(all_task_files, size=min(max_tasks, len(all_task_files)), replace=False)
    return [os.path.splitext(f)[0] for f in selected_files]

class TestREARCDataset(unittest.TestCase):
    
    def setUp(self):
        """Set up for the tests with limited task IDs."""
        self.data_path = RE_ARC_DATA_PATH
        print(f"Using RE-ARC data path: {self.data_path}")
        
        try:
            self.task_ids = get_limited_task_ids(self.data_path, MAX_TEST_TASKS)
            self.assertGreater(len(self.task_ids), 0, "No task IDs found")
            print(f"Testing with {len(self.task_ids)} task IDs: {self.task_ids}")
        except Exception as e:
            self.skipTest(f"Setup failed: {str(e)}")
    
    def test_rearcllm_dataset(self):
        """Tests the REARCLLM Dataset with memory efficiency."""
        try:
            dataset = REARCLLMDataset(
                data_path=self.data_path, 
                split="training",
                task_ids=self.task_ids
            )
            
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
                batch_size=2,  # Small batch size for memory efficiency
                shuffle=True,
                collate_fn=llm_collate_fn
            )
            
            batch = next(iter(dataloader))
            print(f"Batch contains the following keys: {batch.keys()}")
            print(f"Input-IDs Shape: {batch['input_ids'].shape}")
            print(f"Output-IDs Shape: {batch['output_ids'].shape}")
            print(f"Attention-Mask Shape: {batch['attention_mask'].shape}")
        except Exception as e:
            self.fail(f"Test failed: {str(e)}")
    
    def test_rearctask_dataset(self):
        """Tests the REARCTask Dataset with memory efficiency."""
        try:
            dataset = REARCTaskDataset(
                data_path=self.data_path, 
                split="training",
                task_ids=self.task_ids,
                examples_per_task=5  # Limit examples per task for memory efficiency
            )
            
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
                if 'output_grid' in test_example:
                    self.assertIn("output_ids", test_example)
        except Exception as e:
            self.fail(f"Test failed: {str(e)}")
    
    def test_get_task_by_id(self):
        """Tests retrieving a specific task by ID."""
        try:
            dataset = REARCTaskDataset(
                data_path=self.data_path, 
                split="training",
                task_ids=self.task_ids,
                examples_per_task=5
            )
            
            if len(self.task_ids) > 0:
                task_id = self.task_ids[0]
                task = dataset.get_task_by_id(task_id)
                self.assertEqual(task["task_id"], task_id)
                self.assertIn("train", task)
                self.assertIn("test", task)
        except Exception as e:
            self.fail(f"Test failed: {str(e)}")


def test_rearc_task_dataset():
    """Test the REARCTaskDataset class with limited memory usage."""
    try:
        print(f"Using RE-ARC data path: {RE_ARC_DATA_PATH}")
        # Get limited task IDs
        task_ids = get_limited_task_ids(RE_ARC_DATA_PATH, MAX_TEST_TASKS)
        print(f"Testing with task IDs: {task_ids}")
        
        # Training dataset
        train_dataset = REARCTaskDataset(
            data_path=RE_ARC_DATA_PATH,
            split="training",
            task_ids=task_ids,
            examples_per_task=5
        )
        print(f"Number of tasks in the training dataset: {len(train_dataset)}")
        
        if len(train_dataset) == 0:
            print("No tasks in training dataset, skipping further tests")
            return
        
        # Check structure of a task
        task = train_dataset[0]
        print(f"Task structure: {task.keys()}")
        print(f"Task ID: {task['task_id']}")
        print(f"Number of training examples: {len(task['train'])}")
        print(f"Number of test examples: {len(task['test'])}")
        
        # Check structure of a training example
        if task['train']:
            train_example = task['train'][0]
            print(f"Training example structure: {train_example.keys()}")
        
        # Check structure of a test example
        if task['test']:
            test_example = task['test'][0]
            print(f"Test example structure: {test_example.keys()}")
    except Exception as e:
        print(f"Error in test_rearc_task_dataset: {str(e)}")

def test_rearc_dataloader():
    """Test the DataLoader for RE-ARC with memory efficiency."""
    try:
        print(f"Using RE-ARC data path: {RE_ARC_DATA_PATH}")
        # Get limited task IDs
        task_ids = get_limited_task_ids(RE_ARC_DATA_PATH, MAX_TEST_TASKS)
        
        # Create DataLoader for individual examples
        train_dataloader = create_re_arc_dataloader(
            data_path=RE_ARC_DATA_PATH,
            split="training", 
            batch_size=4,
            shuffle=False,
            by_task=False,
            task_ids=task_ids,
            examples_per_task=5
        )
        
        batch = next(iter(train_dataloader))
        print(f"Batch structure: {batch.keys()}")
        print(f"Input-IDs Shape: {batch['input_ids'].shape}")
        print(f"Output-IDs Shape: {batch['output_ids'].shape}")
        print(f"Attention-Mask Shape: {batch['attention_mask'].shape}")
        
        # Create DataLoader for whole tasks
        task_dataloader = create_re_arc_dataloader(
            data_path=RE_ARC_DATA_PATH,
            split="training", 
            batch_size=2,
            shuffle=False,
            by_task=True,
            task_ids=task_ids,
            examples_per_task=5
        )
        
        task_batch = next(iter(task_dataloader))
        print(f"Task batch structure: {task_batch[0].keys() if isinstance(task_batch, list) else 'Dict with keys: ' + str(task_batch.keys())}")
    except Exception as e:
        print(f"Error in test_rearc_dataloader: {str(e)}")

if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    print(f"Test file location: {os.path.abspath(__file__)}")
    print(f"Parent directory: {parent_dir}")
    print(f"RE-ARC data path: {RE_ARC_DATA_PATH}")
    
    # Check if RE-ARC data path exists
    if not os.path.exists(RE_ARC_DATA_PATH):
        print(f"Warning: RE-ARC data path does not exist: {RE_ARC_DATA_PATH}")
    else:
        print(f"RE-ARC data path exists")
        # Check if tasks directory exists
        tasks_dir = os.path.join(RE_ARC_DATA_PATH, "tasks")
        if not os.path.exists(tasks_dir):
            print(f"Warning: Tasks directory does not exist: {tasks_dir}")
        else:
            print(f"Tasks directory exists with {len([f for f in os.listdir(tasks_dir) if f.endswith('.json')])} JSON files")
    
    # Run individual tests to control memory usage
    print("\nRunning test_rearc_task_dataset:")
    test_rearc_task_dataset()
    
    print("\nRunning test_rearc_dataloader:")
    test_rearc_dataloader()
    
    print("\nRunning unittest tests:")
    unittest.main() 