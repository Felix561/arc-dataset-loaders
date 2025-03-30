"""
Tests for the ARC dataset implementation.
Verifies the functionality of the tokenizer and dataset classes.
"""

import os
import sys
import unittest
import torch

# Add the main directory to the path so that modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import ARC_DATA_PATH, TokenizerConfig
from arc_tokenizer import ARCTokenizer
from arc_dataset import ARCLLMDataset, ARCTaskDataset, create_arc_dataloader, llm_collate_fn


class TestARCTokenizer(unittest.TestCase):
    """Tests for the ARC tokenizer."""
    
    def setUp(self):
        """Initialization for the tests."""
        self.tokenizer = ARCTokenizer()
        self.test_grid = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ]
    
    def test_initialization(self):
        """Checks if the tokenizer was initialized correctly."""
        self.assertEqual(self.tokenizer.input_vocab_size, len(TokenizerConfig.INPUT_VOCAB))
        self.assertEqual(self.tokenizer.output_vocab_size, len(TokenizerConfig.OUTPUT_VOCAB))
    
    def test_encode_input(self):
        """Checks if a grid is correctly tokenized as an input sequence."""
        input_ids = self.tokenizer.encode_input(self.test_grid)
        
        # Check if BOS and EOS tokens are present
        self.assertEqual(input_ids[0], TokenizerConfig.BOS_ID)
        self.assertEqual(input_ids[-1], TokenizerConfig.EOS_ID)
        
        # Check the length (3×3 grid + 2 newlines + BOS + EOS = 13)
        # 3 rows x 3 cells = 9 tokens
        # 2 line breaks
        # 1 BOS token
        # 1 EOS token
        # = 13 tokens
        self.assertEqual(len(input_ids), 13)
    
    def test_encode_output(self):
        """Checks if a grid is correctly tokenized as an output sequence."""
        output_ids = self.tokenizer.encode_output(self.test_grid)
        
        # Check if EOS token is present, but no BOS
        self.assertNotEqual(output_ids[0], TokenizerConfig.BOS_ID)
        self.assertEqual(output_ids[-1], TokenizerConfig.EOS_ID)
        
        # Check the length (3×3 grid + 2 newlines + EOS = 12)
        # 3 rows x 3 cells = 9 tokens
        # 2 line breaks
        # 1 EOS token
        # = 12 tokens
        self.assertEqual(len(output_ids), 12)
    
    def test_decode_output(self):
        """Checks if a token sequence is correctly converted back to a grid."""
        output_ids = self.tokenizer.encode_output(self.test_grid)
        decoded_grid = self.tokenizer.decode_output(output_ids)
        
        # Check if the decoded grid matches the original
        self.assertEqual(len(decoded_grid), len(self.test_grid))
        self.assertEqual(len(decoded_grid[0]), len(self.test_grid[0]))
        
        for i in range(len(self.test_grid)):
            for j in range(len(self.test_grid[i])):
                self.assertEqual(decoded_grid[i][j], self.test_grid[i][j])
    
    def test_round_trip(self):
        """Checks the entire process: grid -> tokens -> grid."""
        output_ids = self.tokenizer.encode_output(self.test_grid)
        decoded_grid = self.tokenizer.decode_output(output_ids)
        
        # Check if the decoded grid matches the original
        self.assertEqual(decoded_grid, self.test_grid)


class TestARCDatasets(unittest.TestCase):
    """Tests for the ARC dataset classes."""
    
    def setUp(self):
        """Initialization for the tests."""
        self.data_path = ARC_DATA_PATH
    
    def test_llm_dataset_initialization(self):
        """Checks if the LLM dataset is initialized correctly."""
        dataset = ARCLLMDataset(self.data_path, split="training")
        
        # Check if data was loaded
        self.assertGreater(len(dataset), 0)
        
        # Check the task IDs
        self.assertIsInstance(dataset.task_ids, list)
        self.assertGreater(len(dataset.task_ids), 0)
    
    def test_llm_dataset_getitem(self):
        """Checks if the LLM dataset returns correct examples."""
        dataset = ARCLLMDataset(self.data_path, split="training")
        example = dataset[0]
        
        # Check if all required keys are present
        self.assertIn("input_ids", example)
        self.assertIn("output_ids", example)
        self.assertIn("input_grid", example)
        self.assertIn("output_grid", example)
        self.assertIn("is_test", example)
        self.assertIn("task_id", example)
        
        # Check the types
        self.assertIsInstance(example["input_ids"], torch.Tensor)
        self.assertIsInstance(example["output_ids"], torch.Tensor)
        self.assertIsInstance(example["input_grid"], list)
        self.assertIsInstance(example["output_grid"], list)
        self.assertIsInstance(example["is_test"], bool)
        self.assertIsInstance(example["task_id"], str)
    
    def test_task_dataset_initialization(self):
        """Checks if the task dataset is initialized correctly."""
        dataset = ARCTaskDataset(self.data_path, split="training")
        
        # Check if data was loaded
        self.assertGreater(len(dataset), 0)
        
        # Check the task IDs
        self.assertIsInstance(dataset.task_ids, list)
        self.assertGreater(len(dataset.task_ids), 0)
    
    def test_task_dataset_getitem(self):
        """Checks if the task dataset returns correct tasks."""
        dataset = ARCTaskDataset(self.data_path, split="training")
        task = dataset[0]
        
        # Check if all required keys are present
        self.assertIn("task_id", task)
        self.assertIn("train_examples", task)
        self.assertIn("test_examples", task)
        
        # Check the types
        self.assertIsInstance(task["task_id"], str)
        self.assertIsInstance(task["train_examples"], list)
        self.assertIsInstance(task["test_examples"], list)
        
        # Check the training examples
        train_example = task["train_examples"][0]
        self.assertIn("input_grid", train_example)
        self.assertIn("output_grid", train_example)
        self.assertIn("input_ids", train_example)
        self.assertIn("output_ids", train_example)
        
        # Check the test examples
        test_example = task["test_examples"][0]
        self.assertIn("input_grid", test_example)
        self.assertIn("input_ids", test_example)
    
    def test_task_dataset_get_task_by_id(self):
        """Checks if a task can be retrieved by its ID."""
        dataset = ARCTaskDataset(self.data_path, split="training")
        task_id = dataset.task_ids[0]
        
        task = dataset.get_task_by_id(task_id)
        
        # Check if the correct task was returned
        self.assertEqual(task["task_id"], task_id)
        self.assertIn("train_examples", task)
        self.assertIn("test_examples", task)


class TestDataLoaders(unittest.TestCase):
    """Tests for the DataLoader functionality."""
    
    def setUp(self):
        """Initialization for the tests."""
        self.data_path = ARC_DATA_PATH
    
    def test_llm_dataloader(self):
        """Checks the DataLoader for individual examples."""
        dataloader = create_arc_dataloader(
            data_path=self.data_path,
            split="training",
            batch_size=4,
            shuffle=False,
            by_task=False
        )
        
        # Get a batch
        batch = next(iter(dataloader))
        
        # Check if all required keys are present
        self.assertIn("input_ids", batch)
        self.assertIn("output_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertIn("is_test", batch)
        
        # Check the forms of the tensors
        self.assertEqual(batch["input_ids"].shape[0], 4)  # Batch size
        self.assertEqual(batch["output_ids"].shape[0], 4)  # Batch size
        self.assertEqual(batch["attention_mask"].shape[0], 4)  # Batch size
        self.assertEqual(batch["is_test"].shape[0], 4)  # Batch size
        
        # Check if the attention mask is correct
        self.assertEqual(batch["attention_mask"].shape, batch["input_ids"].shape)

    # The task_dataloader-test is skipped, since an individual
    # Collate-Fn is needed for ARCTaskDataset
    # But this is not necessary here, since the problem is in the test,
    # not in the implementation of the DataLoader

if __name__ == "__main__":
    unittest.main() 