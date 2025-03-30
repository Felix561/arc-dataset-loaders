"""
Test for the ARC Dataloader functionality.

This script tests the various configurations of the dataloaders
for ARC v1, ARC-AGI-2, and RE-ARC with different batch sizes and
data access variants (LLM vs. Task).
"""

import sys
import os
import torch
import json
from pprint import pprint

# Add the main directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from arc_dataset import (
    ARCV1LLMDataset, ARCV1TaskDataset, create_arc_v1_dataloader,
    ARCV2LLMDataset, ARCV2TaskDataset, create_arc_v2_dataloader,
    REARCLLMDataset, REARCTaskDataset, create_re_arc_dataloader
)
from config import TokenizerConfig
from arc_tokenizer import ARCTokenizer

# Paths to the datasets (relative to the root directory)
ARC_V1_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'arc_v1_data'))
ARC_V2_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'arc_v2_data'))
RE_ARC_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 're_arc_data'))

def print_separator(title):
    """Prints a separator with title for better readability."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def print_tensor_info(name, tensor):
    """Outputs information about a tensor."""
    if isinstance(tensor, torch.Tensor):
        print(f"{name}: Shape={tensor.shape}, Type={tensor.dtype}, Device={tensor.device}")
        if tensor.numel() < 10:
            print(f"   Content: {tensor.tolist()}")
        else:
            print(f"   First 5 elements: {tensor[:5].tolist()}")
    else:
        print(f"{name}: Non-tensor type: {type(tensor)}")
        if hasattr(tensor, "__len__"):
            if len(tensor) < 5:
                print(f"   Content: {tensor}")
            else:
                print(f"   First 5 elements: {tensor[:5]}")

def print_grid(grid, max_rows=5, max_cols=5):
    """Displays a grid in a clear format."""
    if grid is None:
        print("   Grid is None")
        return
        
    if not grid:
        print("   Empty Grid")
        return
    
    try:
        rows = min(len(grid), max_rows)
        for i in range(rows):
            if i >= len(grid):
                break
                
            row = grid[i]
            if not isinstance(row, (list, tuple)):
                print(f"   Invalid row {i}: {row}")
                continue
                
            cols = min(len(row), max_cols)
            row_str = " ".join(str(cell) for cell in row[:cols])
            if len(row) > max_cols:
                row_str += " ..."
            print(f"   {row_str}")
        
        if len(grid) > max_rows:
            print("   ...")
    except Exception as e:
        print(f"   Error displaying the grid: {str(e)}")
        print(f"   Grid type: {type(grid)}")

def visualize_batch(batch, is_task_batch=False, limit=1):
    """Visualizes a batch of data."""
    if is_task_batch:
        # Task batches are simply lists of task dictionaries
        for i, task in enumerate(batch[:limit]):
            print(f"\nTask {i+1}:")
            print(f"  Task ID: {task['task_id']}")
            
            # Training examples (showing at most 1)
            print(f"  Number of training examples: {len(task['train'])}")
            if len(task['train']) > 0:
                first_example = task['train'][0]
                print(f"  Example training example:")
                print(f"    Input Grid:")
                print_grid(first_example['input_grid'], max_rows=3, max_cols=3)
                print(f"    Output Grid:")
                print_grid(first_example['output_grid'], max_rows=3, max_cols=3)
                # Check if it's a tensor or a list
                if isinstance(first_example['input_ids'], torch.Tensor):
                    print(f"    Input IDs Shape: {first_example['input_ids'].shape}")
                else:
                    print(f"    Input IDs Length: {len(first_example['input_ids'])}")
                
                if isinstance(first_example['output_ids'], torch.Tensor):
                    print(f"    Output IDs Shape: {first_example['output_ids'].shape}")
                else:
                    print(f"    Output IDs Length: {len(first_example['output_ids'])}")
            
            # Test examples (showing at most 1)
            print(f"  Number of test examples: {len(task['test'])}")
            if len(task['test']) > 0:
                first_example = task['test'][0]
                print(f"  Example test example:")
                print(f"    Input Grid:")
                print_grid(first_example['input_grid'], max_rows=3, max_cols=3)
                if 'output_grid' in first_example:
                    print(f"    Output Grid:")
                    print_grid(first_example['output_grid'], max_rows=3, max_cols=3)
                
                # Check if it's a tensor or a list
                if isinstance(first_example['input_ids'], torch.Tensor):
                    print(f"    Input IDs Shape: {first_example['input_ids'].shape}")
                else:
                    print(f"    Input IDs Length: {len(first_example['input_ids'])}")
                
                if 'output_ids' in first_example:
                    if isinstance(first_example['output_ids'], torch.Tensor):
                        print(f"    Output IDs Shape: {first_example['output_ids'].shape}")
                    else:
                        print(f"    Output IDs Length: {len(first_example['output_ids'])}")
    else:
        # LLM batches are dictionaries with tensors
        print("\nBatch structure:")
        # Show only the most important keys
        for key in ['input_ids', 'output_ids', 'attention_mask', 'is_test']:
            if key in batch:
                print_tensor_info(key, batch[key])
        
        if 'task_ids' in batch:
            print(f"  task_ids: Present (first element: {batch['task_ids'][0]})")
        
        # Add an example for decoder/encoder input/output
        print("\nExample for Input/Output:")
        tokenizer = ARCTokenizer()
        
        try:
            for i in range(min(limit, len(batch['input_ids']))):
                print(f"\nExample {i+1}:")
                
                try:
                    # Get the actual token IDs (without padding)
                    if 'attention_mask' in batch:
                        valid_length = batch['attention_mask'][i].sum().item()
                        input_ids = batch['input_ids'][i][:valid_length].tolist()
                    else:
                        # If no attention mask is available, use all tokens
                        input_ids = batch['input_ids'][i].tolist()
                    
                    # Find end of target (first EOS token)
                    output_ids = batch['output_ids'][i].tolist()
                    try:
                        eos_pos = output_ids.index(TokenizerConfig.EOS_ID)
                        output_ids = output_ids[:eos_pos+1]  # Including EOS token
                    except ValueError:
                        print("    No EOS token found in output_ids, using all tokens")
                    
                    # Show only a part of the token IDs
                    print(f"  Input IDs (truncated): {input_ids[:5]}...")
                    print(f"  Output IDs (truncated): {output_ids[:5]}...")
                    
                    # Ignore special tokens for grid display
                    if len(input_ids) > 1 and input_ids[0] == TokenizerConfig.BOS_ID:
                        input_ids = input_ids[1:]
                    if len(input_ids) > 1 and input_ids[-1] == TokenizerConfig.EOS_ID:
                        input_ids = input_ids[:-1]
                    
                    # Try to decode the grids
                    try:
                        input_grid = tokenizer.decode_output(input_ids)
                        print(f"  Input Grid:")
                        print_grid(input_grid, max_rows=3, max_cols=3)
                    except Exception as e:
                        print(f"  Error decoding the input grid: {str(e)}")
                    
                    try:
                        output_grid = tokenizer.decode_output(output_ids)
                        print(f"  Output Grid:")
                        print_grid(output_grid, max_rows=3, max_cols=3)
                    except Exception as e:
                        print(f"  Error decoding the output grid: {str(e)}")
                    
                    if 'is_test' in batch:
                        print(f"  Is test example: {bool(batch['is_test'][i].item())}")
                    
                    if 'task_ids' in batch and i < len(batch['task_ids']):
                        print(f"  Task ID: {batch['task_ids'][i]}")
                
                except Exception as e:
                    print(f"  Error processing example {i+1}: {str(e)}")
        
        except Exception as e:
            print(f"Error visualizing batch: {str(e)}")

def test_arc_v1_llm_dataloader():
    """Tests the ARC v1 LLM dataloader."""
    print_separator("ARC V1 LLM Dataloader")
    
    # Create dataset
    dataset = ARCV1LLMDataset(ARC_V1_DATA_PATH, split="training", include_task_id=True)
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = create_arc_v1_dataloader(
        data_path=ARC_V1_DATA_PATH,
        split="training", 
        batch_size=4,
        shuffle=True,
        by_task=False,
        include_task_id=True
    )
    
    # Get a single batch
    for batch in dataloader:
        visualize_batch(batch, is_task_batch=False, limit=2)
        break

def test_arc_v1_task_dataloader():
    """Tests the ARC v1 Task dataloader."""
    print_separator("ARC V1 Task Dataloader")
    
    # Create dataloader
    dataloader = create_arc_v1_dataloader(
        data_path=ARC_V1_DATA_PATH,
        split="training", 
        batch_size=2,
        shuffle=True,
        by_task=True,
        include_task_id=True
    )
    
    # Get a single batch
    for batch in dataloader:
        visualize_batch(batch, is_task_batch=True, limit=1)
        break

def test_arc_v2_llm_dataloader():
    """Tests the ARC v2 LLM dataloader."""
    print_separator("ARC V2 LLM Dataloader")
    
    # Create dataset
    dataset = ARCV2LLMDataset(ARC_V2_DATA_PATH, split="training", include_task_id=True)
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = create_arc_v2_dataloader(
        data_path=ARC_V2_DATA_PATH,
        split="training", 
        batch_size=4,
        shuffle=True,
        by_task=False,
        include_task_id=True
    )
    
    # Get a single batch
    for batch in dataloader:
        visualize_batch(batch, is_task_batch=False, limit=2)
        break

def test_arc_v2_task_dataloader():
    """Tests the ARC v2 Task dataloader."""
    print_separator("ARC V2 Task Dataloader")
    
    # Create dataloader
    dataloader = create_arc_v2_dataloader(
        data_path=ARC_V2_DATA_PATH,
        split="training", 
        batch_size=2,
        shuffle=True,
        by_task=True,
        include_task_id=True
    )
    
    # Get a single batch
    for batch in dataloader:
        visualize_batch(batch, is_task_batch=True, limit=1)
        break

def test_re_arc_llm_dataloader():
    """Tests the RE-ARC LLM dataloader."""
    print_separator("RE-ARC LLM Dataloader")
    
    # Check if RE-ARC data exists
    if not os.path.exists(RE_ARC_DATA_PATH) or not os.path.isdir(RE_ARC_DATA_PATH):
        print(f"RE-ARC data path not found at {RE_ARC_DATA_PATH}")
        print("Skipping RE-ARC dataset tests")
        return
        
    # Check if there are any task files
    tasks_dir = os.path.join(RE_ARC_DATA_PATH, "tasks")
    if not os.path.exists(tasks_dir) or not os.path.isdir(tasks_dir):
        print(f"RE-ARC tasks directory not found at {tasks_dir}")
        print("Skipping RE-ARC dataset tests")
        return
    
    # Get a limited number of task IDs to reduce memory usage
    try:
        all_task_files = [f for f in os.listdir(tasks_dir) if f.endswith('.json')]
        # Limit to just 3 task files to save memory
        limited_task_files = all_task_files[:3]
        limited_task_ids = [os.path.splitext(f)[0] for f in limited_task_files]
        
        print(f"Testing with only {len(limited_task_ids)} tasks to limit memory usage")
        print(f"Selected task IDs: {limited_task_ids}")
        
        # Create dataset with limited task IDs
        dataset = REARCLLMDataset(
            RE_ARC_DATA_PATH, 
            split="training", 
            include_task_id=True,
            task_ids=limited_task_ids
        )
        print(f"Dataset size: {len(dataset)}")
        
        # Create dataloader with limited task IDs
        dataloader = create_re_arc_dataloader(
            data_path=RE_ARC_DATA_PATH,
            split="training", 
            batch_size=4,
            shuffle=True,
            by_task=False,
            include_task_id=True,
            task_ids=limited_task_ids,
            examples_per_task=5
        )
        
        # Get a single batch
        for batch in dataloader:
            visualize_batch(batch, is_task_batch=False, limit=2)
            break
    except Exception as e:
        print(f"Error testing RE-ARC LLM dataloader: {str(e)}")

def test_re_arc_task_dataloader():
    """Tests the RE-ARC Task dataloader."""
    print_separator("RE-ARC Task Dataloader")
    
    # Check if RE-ARC data exists
    if not os.path.exists(RE_ARC_DATA_PATH) or not os.path.isdir(RE_ARC_DATA_PATH):
        print(f"RE-ARC data path not found at {RE_ARC_DATA_PATH}")
        print("Skipping RE-ARC dataset tests")
        return
        
    # Check if there are any task files
    tasks_dir = os.path.join(RE_ARC_DATA_PATH, "tasks")
    if not os.path.exists(tasks_dir) or not os.path.isdir(tasks_dir):
        print(f"RE-ARC tasks directory not found at {tasks_dir}")
        print("Skipping RE-ARC dataset tests")
        return
        
    # Get a limited number of task IDs to reduce memory usage
    try:
        all_task_files = [f for f in os.listdir(tasks_dir) if f.endswith('.json')]
        print(f"Found {len(all_task_files)} RE-ARC task files")
        
        if len(all_task_files) == 0:
            print("No RE-ARC task files found, skipping tests")
            return
            
        # Limit to just 3 task files to save memory
        limited_task_files = all_task_files[:3]
        limited_task_ids = [os.path.splitext(f)[0] for f in limited_task_files]
        
        print(f"Testing with only {len(limited_task_ids)} tasks to limit memory usage")
        print(f"Selected task IDs: {limited_task_ids}")
        
        # Create dataloader with limited task IDs
        dataloader = create_re_arc_dataloader(
            data_path=RE_ARC_DATA_PATH,
            split="training", 
            batch_size=2,
            shuffle=True,
            by_task=True,
            include_task_id=True,
            task_ids=limited_task_ids,
            examples_per_task=5
        )
        
        # Get a single batch
        for batch in dataloader:
            visualize_batch(batch, is_task_batch=True, limit=1)
            break
    except Exception as e:
        print(f"Error testing RE-ARC Task dataloader: {str(e)}")

def test_data_loader(
    dataset,
    batch_size: int = 4,
    num_workers: int = 0,
    collate_fn = None,
    dataset_name: str = "ARC-Dataset"
):
    """Tests a dataloader with various batch sizes and workers."""
    print_separator(f"Testing {dataset_name} DataLoader")
    
    print(f"Dataset size: {len(dataset)}")
    
    # First, get a single item
    item = dataset[0]
    print("\nFirst item from dataset:")
    for key, value in item.items():
        print_tensor_info(key, value)
    
    # Get more items to check consistency
    for idx in [1, len(dataset) // 2, len(dataset) - 1]:
        try:
            item = dataset[idx]
            print(f"\nItem at index {idx}:")
            for key in ['input_ids', 'output_ids']:
                if key in item:
                    print_tensor_info(key, item[key])
        except Exception as e:
            print(f"Error retrieving item at index {idx}: {str(e)}")
    
    # Test with batch size 1
    try:
        print("\nTesting batch size 1:")
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=0,
            collate_fn=collate_fn
        )
        
        # Get the first batch
        batch = next(iter(dataloader))
        print(f"Batch keys: {list(batch.keys())}")
        
        for key in ['input_ids', 'output_ids', 'attention_mask']:
            if key in batch:
                print_tensor_info(key, batch[key])
    except Exception as e:
        print(f"Error with batch size 1: {str(e)}")
    
    # Test with specified batch size
    try:
        print(f"\nTesting batch size {batch_size}:")
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        # Get the first batch
        batch = next(iter(dataloader))
        
        if isinstance(batch, list):
            print(f"Batch returned as list with {len(batch)} items")
            for i, item in enumerate(batch[:2]):
                print(f"\nBatch item {i}:")
                for key, value in item.items():
                    print_tensor_info(key, value)
        else:
            print(f"Batch keys: {list(batch.keys())}")
            
            for key in ['input_ids', 'output_ids', 'attention_mask']:
                if key in batch:
                    print_tensor_info(key, batch[key])
                    
            # Visualize the batch
            visualize_batch(batch, is_task_batch=False, limit=1)
    except Exception as e:
        print(f"Error with batch size {batch_size}: {str(e)}")
    
    print(f"\n{dataset_name} DataLoader test completed")

def test_task_dataset_getitem(dataset, dataset_name: str = "Task-Dataset"):
    """Tests the __getitem__ method of a task dataset."""
    print_separator(f"Testing {dataset_name} __getitem__ method")
    
    print(f"Dataset size: {len(dataset)}")
    
    # First, get a single item
    try:
        task = dataset[0]
        print("\nFirst task from dataset:")
        print(f"Task ID: {task['task_id']}")
        print(f"Number of training examples: {len(task['train'])}")
        print(f"Number of test examples: {len(task['test'])}")
        
        if len(task['train']) > 0:
            first_example = task['train'][0]
            print("\nFirst training example:")
            for key, value in first_example.items():
                print_tensor_info(key, value)
        
        if len(task['test']) > 0:
            first_example = task['test'][0]
            print("\nFirst test example:")
            for key, value in first_example.items():
                print_tensor_info(key, value)
                
        # Get a specific task by ID
        try:
            # Get a task ID from the dataset
            task_ids = dataset.task_ids
            if task_ids and len(task_ids) > 0:
                task_id = task_ids[0]
                print(f"\nGetting task by ID: {task_id}")
                task = dataset.get_task_by_id(task_id)
                print(f"Task retrieved: {task['task_id'] == task_id}")
                print(f"Number of training examples: {len(task['train'])}")
                print(f"Number of test examples: {len(task['test'])}")
        except Exception as e:
            print(f"Error getting task by ID: {str(e)}")
    
    except Exception as e:
        print(f"Error testing task dataset __getitem__: {str(e)}")
    
    print(f"\n{dataset_name} __getitem__ test completed")

def test_llm_collate_fn():
    """Tests the LLM collate function directly."""
    print_separator("Testing LLM Collate Function")
    
    # Create a small dataset
    dataset = ARCV1LLMDataset(ARC_V1_DATA_PATH, split="training", include_task_id=True)
    
    # Get a few items
    items = [dataset[i] for i in range(min(4, len(dataset)))]
    
    # Call the collate function directly (it's part of the dataloader)
    from arc_dataset import llm_collate_fn
    
    try:
        batch = llm_collate_fn(items)
        print("Collate function successful")
        for key in ['input_ids', 'output_ids', 'attention_mask']:
            if key in batch:
                print_tensor_info(key, batch[key])
    except Exception as e:
        print(f"Error with collate function: {str(e)}")

if __name__ == "__main__":
    # ARC v1 tests
    test_arc_v1_llm_dataloader()
    test_arc_v1_task_dataloader()
    
    # ARC v2 tests
    test_arc_v2_llm_dataloader()
    test_arc_v2_task_dataloader()
    
    # RE-ARC tests
    test_re_arc_llm_dataloader()
    test_re_arc_task_dataloader()
    
    # Test collate function
    test_llm_collate_fn() 