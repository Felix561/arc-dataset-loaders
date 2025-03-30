"""
Utility functions for the ARC dataset.
"""

import json
import os
from typing import Dict, List, Any, Optional, Union
import torch
from arc_tokenizer import ARCTokenizer

def save_predictions_to_json(
    predictions: Dict[str, Union[List[List[List[int]]], List[List[List[List[int]]]]]],
    output_file: str,
    attempts_per_task: int = 2,
    duplicate_if_fewer_attempts: bool = True
) -> None:
    """
    Saves the model predictions to a JSON file in Kaggle submission format.
    
    Args:
        predictions: A dictionary with task IDs as keys and lists of predicted grids as values.
                    Format 1 (Single attempts): {'task_id': [grid1, grid2, ...]} 
                    where each grid is a list of lists of integers.
                    Format 2 (Multiple attempts): {'task_id': [[grid1_attempt1, grid1_attempt2], [grid2_attempt1, grid2_attempt2], ...]}
                    where each grid is a list of lists of integers.
        output_file: Path to the output file
        attempts_per_task: Number of attempts per task/test input (default 2)
        duplicate_if_fewer_attempts: If True, missing attempts are filled by duplicating the last attempt
                                    If False, only the actually available attempts are used
        
    Returns:
        None, the results are written to the specified file
    """
    # Format: {task_id: [{attempt_1: grid, attempt_2: grid}, ...]}
    formatted_predictions = {}
    
    for task_id, grids_list in predictions.items():
        task_predictions = []
        
        # For each test input in this task
        for test_idx, grid in enumerate(grids_list):
            # Check if this test input already contains multiple attempts
            if isinstance(grid, list):
                if not grid:
                    # Empty grid for this test input
                    attempts = {}
                    for i in range(1, attempts_per_task + 1):
                        attempts[f"attempt_{i}"] = []
                    task_predictions.append(attempts)
                elif isinstance(grid[0], list) and (not grid[0] or isinstance(grid[0][0], int)):
                    # Simple grid for this test input (one attempt)
                    attempts = {}
                    
                    if duplicate_if_fewer_attempts:
                        # Duplicate the attempt for all requested attempts
                        for i in range(1, attempts_per_task + 1):
                            attempts[f"attempt_{i}"] = grid
                    else:
                        # Only one attempt without duplication
                        attempts["attempt_1"] = grid
                        
                    task_predictions.append(attempts)
                elif isinstance(grid[0], list) and isinstance(grid[0][0], list):
                    # Multiple grids for this test input (multiple attempts)
                    attempts = {}
                    
                    # Save each attempt
                    for i in range(1, attempts_per_task + 1):
                        if i <= len(grid):
                            attempts[f"attempt_{i}"] = grid[i-1]
                        elif duplicate_if_fewer_attempts and grid:
                            # If not enough attempts and duplication is enabled
                            attempts[f"attempt_{i}"] = grid[-1]
                    
                    task_predictions.append(attempts)
                else:
                    # Invalid grid type
                    attempts = {}
                    for i in range(1, attempts_per_task + 1):
                        attempts[f"attempt_{i}"] = [[0, 0]]  # Fallback
                    task_predictions.append(attempts)
            else:
                # Invalid grid type
                attempts = {}
                for i in range(1, attempts_per_task + 1):
                    attempts[f"attempt_{i}"] = [[0, 0]]  # Fallback
                task_predictions.append(attempts)
        
        formatted_predictions[task_id] = task_predictions
    
    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(formatted_predictions, f)

def predictions_from_model_outputs(
    task_ids: List[str],
    generated_ids: torch.Tensor,
    tokenizer: ARCTokenizer,
    test_inputs_per_task: Optional[Dict[str, int]] = None
) -> Dict[str, List[List[List[int]]]]:
    """
    Converts the generated token IDs from the model back to a dictionary with task IDs and grids.
    
    Args:
        task_ids: List of task IDs in the same order as the generated outputs
        generated_ids: Tensor with the generated token IDs from the model (batch_size, sequence_length)
        tokenizer: ARCTokenizer instance for decoding the token IDs
        test_inputs_per_task: Dictionary with task IDs as keys and the number of test inputs as values.
                             If not specified, it is assumed that each task has exactly one test input.
    
    Returns:
        Dictionary in the format {task_id: [grid1, grid2, ...]} for all tasks
    """
    predictions = {}
    
    if test_inputs_per_task is None:
        # If no information about test inputs is available, assume there is one per task
        test_inputs_per_task = {task_id: 1 for task_id in task_ids}
    
    # Initialize the dictionary with empty lists for all tasks
    for task_id in set(task_ids):
        predictions[task_id] = []
    
    # Create a mapping from task ID to position in the batch
    task_id_to_indices = {}
    for idx, task_id in enumerate(task_ids):
        if task_id not in task_id_to_indices:
            task_id_to_indices[task_id] = []
        task_id_to_indices[task_id].append(idx)
    
    # Process each task separately
    for task_id, indices in task_id_to_indices.items():
        task_grids = []
        num_test_inputs = min(len(indices), test_inputs_per_task.get(task_id, 1))
        
        # Process each test input for this task
        for i in range(num_test_inputs):
            if i < len(indices) and indices[i] < len(generated_ids):
                # Decode the generated token IDs to a grid
                output_ids = generated_ids[indices[i]].cpu().tolist()
                grid = tokenizer.decode_output(output_ids)
                task_grids.append(grid)
        
        predictions[task_id] = task_grids
    
    return predictions

def load_task_metadata(data_path: str, split: str = "evaluation") -> Dict[str, int]:
    """
    Loads metadata about the tasks, especially the number of test inputs per task.
    
    Args:
        data_path: Path to the ARC dataset folder
        split: Either "training", "evaluation", or "test"
        
    Returns:
        Dictionary with task IDs as keys and the number of test inputs as values
    """
    if split.lower() == "training":
        challenges_file = os.path.join(data_path, "arc-agi_training_challenges.json")
    elif split.lower() == "evaluation":
        challenges_file = os.path.join(data_path, "arc-agi_evaluation_challenges.json")
    else:  # test
        challenges_file = os.path.join(data_path, "arc-agi_test_challenges.json")
    
    with open(challenges_file, 'r') as f:
        challenges = json.load(f)
    
    # Count the number of test inputs per task
    test_inputs_per_task = {}
    for task_id, task in challenges.items():
        test_inputs_per_task[task_id] = len(task.get("test", []))
    
    return test_inputs_per_task

def create_submission_file(
    eval_dataloader, 
    model,
    tokenizer: ARCTokenizer,
    output_file: str,
    device: torch.device,
    max_length: int = 100,
    attempts_per_task: int = 2,
    duplicate_if_fewer_attempts: bool = True,
    num_beams: int = 1
) -> None:
    """
    Creates a complete submission file for the Kaggle competition.
    
    Args:
        eval_dataloader: DataLoader with the evaluation data
        model: The trained model
        tokenizer: ARCTokenizer instance
        output_file: Path to the output file
        device: Torch device (CPU or GPU)
        max_length: Maximum length of the generated sequence
        attempts_per_task: Number of attempts per task (default 2 for Kaggle)
        duplicate_if_fewer_attempts: If True, missing attempts are filled by duplication
        num_beams: Number of beams for beam search decoding (1 = greedy decoding)
    
    Returns:
        None, the submission file is generated at the specified path
    """
    model.eval()
    all_task_ids = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            task_ids = batch.get("task_ids", [])
            
            # Generate predictions
            if hasattr(model, "generate") and num_beams > 1:
                # For Transformer models with generate method and Beam-Search
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=num_beams,
                    num_return_sequences=min(num_beams, attempts_per_task)
                )
                
                # Organize the generated sequences by batch element
                batch_size = input_ids.shape[0]
                num_return_sequences = min(num_beams, attempts_per_task)
                
                # Save task IDs only once per batch element
                for i in range(batch_size):
                    if i < len(task_ids):
                        all_task_ids.append(task_ids[i])
                
                # Now organize the outputs by batch element with multiple attempts per element
                for i in range(0, len(outputs), num_return_sequences):
                    batch_outputs = []
                    for j in range(num_return_sequences):
                        if i+j < len(outputs):
                            batch_outputs.append(outputs[i+j])
                    if batch_outputs:
                        all_predictions.append(torch.stack(batch_outputs))
            
            elif hasattr(model, "generate"):
                # For Transformer models with generate method (without Beam-Search)
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length
                )
                
                # Save task IDs and predictions
                all_task_ids.extend(task_ids)
                for output in outputs:
                    all_predictions.append(output)
            else:
                # For simpler models
                outputs = model(input_ids, attention_mask=attention_mask)
                # Choose the most probable token
                outputs = outputs.argmax(dim=-1)
                
                # Save task IDs and predictions
                all_task_ids.extend(task_ids)
                for output in outputs:
                    all_predictions.append(output)
    
    # Convert all predictions to grids
    if num_beams > 1 and hasattr(model, "generate"):
        # When Beam-Search, predictions are already in the form of a list of lists
        predictions_dict = {}
        for idx, task_id in enumerate(all_task_ids):
            if task_id not in predictions_dict:
                predictions_dict[task_id] = []
            
            # Decode each sequence for this task
            if idx < len(all_predictions):
                batch_grids = []
                for seq in all_predictions[idx]:
                    grid = tokenizer.decode_output(seq.cpu().tolist())
                    batch_grids.append(grid)
                predictions_dict[task_id].append(batch_grids)
    else:
        # When Greedy Search/simple models normal conversion
        predictions_tensor = torch.stack(all_predictions) if all_predictions else torch.tensor([])
        
        # Load metadata for test inputs
        data_path = eval_dataloader.dataset.data_path
        split = eval_dataloader.dataset.split
        test_inputs_per_task = load_task_metadata(data_path, split)
        
        # Convert to dictionary
        predictions_dict = predictions_from_model_outputs(
            all_task_ids, 
            predictions_tensor, 
            tokenizer,
            test_inputs_per_task
        )
    
    # Save in submission format
    save_predictions_to_json(
        predictions_dict,
        output_file,
        attempts_per_task,
        duplicate_if_fewer_attempts
    )
    
    print(f"Submission file created successfully: {output_file}")


# Example for usage
if __name__ == "__main__":
    # Example grids
    grid1 = [[0, 1], [2, 3], [4, 5]]
    grid2 = [[6, 7]]
    grid3 = [[8, 9]]
    grid4 = [[1, 1], [2, 2]]
    
    # Example 1: Simple Predictions (one attempt per test input)
    predictions_simple = {
        "task1": [grid1],
        "task2": [grid2, grid3]  
    }
    
    # Save in submission format with duplication
    save_predictions_to_json(
        predictions=predictions_simple, 
        output_file="example_submission_simple.json",
        attempts_per_task=2,
        duplicate_if_fewer_attempts=True
    )
    
    # Example 2: Multiple attempts per test input
    predictions_multiple = {
        "task1": [[grid1, grid4]],  # One test input with two attempts
        "task2": [[grid2, grid3]]   # One test input with two attempts
    }
    
    # Save in submission format without duplication
    save_predictions_to_json(
        predictions=predictions_multiple, 
        output_file="example_submission_multiple.json",
        attempts_per_task=2,
        duplicate_if_fewer_attempts=False
    )
    
    print("Example submissions created:")
    print("1. example_submission_simple.json - One attempt per test input, duplicated")
    print("2. example_submission_multiple.json - Two attempts per test input, no duplication") 