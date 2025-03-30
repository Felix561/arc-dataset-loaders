"""
Tokenizer for ARC Grids.
Converts 2D grids into token sequences and vice versa.
"""

import numpy as np
from typing import List, Dict, Union, Tuple
from config import TokenizerConfig

class ARCTokenizer:
    """
    Tokenizer for ARC Grids.
    
    Converts 2D grids (lists of lists) into token sequences and vice versa.
    Provides separate tokenizers for input and output with optimized vocabularies.
    """
    
    def __init__(self):
        # Create vocabularies and mapping dictionaries
        self._setup_vocab()
    
    def _setup_vocab(self):
        """Initialize vocabulary mappings for input and output."""
        # Input vocabulary
        self.input_token2id = {token: idx for idx, token in enumerate(TokenizerConfig.INPUT_VOCAB)}
        self.input_id2token = {idx: token for token, idx in self.input_token2id.items()}
        
        # Output vocabulary
        self.output_token2id = {token: idx for idx, token in enumerate(TokenizerConfig.OUTPUT_VOCAB)}
        self.output_id2token = {idx: token for token, idx in self.output_token2id.items()}
        
        # Ensure that important token IDs are directly available
        self.pad_id = self.output_token2id[TokenizerConfig.PAD_TOKEN]
        self.eos_id = self.output_token2id[TokenizerConfig.EOS_TOKEN]
        self.newline_id = self.output_token2id[TokenizerConfig.NEWLINE_TOKEN]
    
    def encode_input(self, grid: List[List[int]]) -> List[int]:
        """
        Convert a grid into a token ID sequence for input.
        
        Args:
            grid: 2D grid as a list of lists with integers (0-9)
            
        Returns:
            List of token IDs with BOS and EOS
        """
        # Convert grid to string tokens
        tokens = []
        tokens.append(TokenizerConfig.BOS_TOKEN)
        
        for i, row in enumerate(grid):
            # Add color tokens
            for cell in row:
                tokens.append(str(cell))
            
            # Add line break, except for the last line
            if i < len(grid) - 1:
                tokens.append(TokenizerConfig.NEWLINE_TOKEN)
        
        tokens.append(TokenizerConfig.EOS_TOKEN)
        
        # Convert tokens to IDs
        token_ids = [self.input_token2id[token] for token in tokens]
        
        return token_ids
    
    def encode_output(self, grid: List[List[int]]) -> List[int]:
        """
        Convert a grid into a token ID sequence for output.
        
        Args:
            grid: 2D grid as a list of lists with integers (0-9)
            
        Returns:
            List of token IDs with EOS (no BOS)
        """
        # Convert grid to string tokens
        tokens = []
        
        for i, row in enumerate(grid):
            # Add color tokens
            for cell in row:
                tokens.append(str(cell))
            
            # Add line break, except for the last line
            if i < len(grid) - 1:
                tokens.append(TokenizerConfig.NEWLINE_TOKEN)
        
        # Important: Explicitly add the EOS token
        tokens.append(TokenizerConfig.EOS_TOKEN)
        
        # Debug output: Check if EOS token was added correctly
        # print(f"Tokens before conversion: {tokens}")
        # print(f"EOS token in output vocabulary: {TokenizerConfig.EOS_TOKEN in self.output_token2id}")
        # print(f"EOS token ID in output vocabulary: {self.output_token2id.get(TokenizerConfig.EOS_TOKEN)}")
        
        # Convert tokens to IDs and ensure that the last token is the EOS ID
        token_ids = []
        for token in tokens:
            if token == TokenizerConfig.EOS_TOKEN:
                token_ids.append(TokenizerConfig.EOS_ID)  # Explicit use of the constant ID
            else:
                token_ids.append(self.output_token2id[token])
        
        # Check the last token
        assert token_ids[-1] == TokenizerConfig.EOS_ID, f"The last token should be EOS, but is {token_ids[-1]}"
        
        return token_ids
    
    def decode_output(self, token_ids: List[int]) -> List[List[int]]:
        """
        Convert a token ID sequence back into a grid.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            2D grid as a list of lists with integers (0-9)
        """
        # Convert IDs to tokens
        tokens = [self.output_id2token[idx] for idx in token_ids]
        
        # Remove EOS token if present
        if TokenizerConfig.EOS_TOKEN in tokens:
            tokens = tokens[:tokens.index(TokenizerConfig.EOS_TOKEN)]
        
        # Convert token sequence back to 2D grid
        grid = []
        current_row = []
        
        for token in tokens:
            if token == TokenizerConfig.NEWLINE_TOKEN:
                grid.append(current_row)
                current_row = []
            elif token in TokenizerConfig.COLOR_TOKENS:
                current_row.append(int(token))
            # Ignore other tokens (like PAD)
        
        # Add the last row if it's not empty
        if current_row:
            grid.append(current_row)
        
        return grid
    
    def grid_to_string(self, grid: List[List[int]]) -> str:
        """
        Convert a grid into a readable string.
        
        Args:
            grid: 2D grid as a list of lists with integers (0-9)
            
        Returns:
            String representation of the grid
        """
        return '\n'.join([' '.join([str(cell) for cell in row]) for row in grid])
    
    @property
    def input_vocab_size(self) -> int:
        """Size of the input vocabulary."""
        return len(self.input_token2id)
    
    @property
    def output_vocab_size(self) -> int:
        """Size of the output vocabulary."""
        return len(self.output_token2id)
    
    def pad_sequence(self, sequence: List[int], max_length: int, is_input: bool = True) -> List[int]:
        """
        Add padding to a sequence to reach max_length.
        
        Args:
            sequence: List of token IDs
            max_length: Target length
            is_input: Whether this is an input sequence
            
        Returns:
            Padded sequence
        """
        pad_id = TokenizerConfig.PAD_ID
        return sequence + [pad_id] * (max_length - len(sequence))


# Example usage
if __name__ == "__main__":
    # Example grid
    grid = [
        [0, 7, 7],
        [7, 7, 7],
        [0, 7, 7]
    ]
    
    tokenizer = ARCTokenizer()
    
    # Tokenize input
    input_ids = tokenizer.encode_input(grid)
    print(f"Input Tokens ({len(input_ids)}): {input_ids}")
    
    # Tokenize output
    output_ids = tokenizer.encode_output(grid)
    print(f"Output Tokens ({len(output_ids)}): {output_ids}")
    print(f"Last token (should be EOS): {output_ids[-1]} (EOS_ID is {TokenizerConfig.EOS_ID})")
    
    # Detokenize
    decoded_grid = tokenizer.decode_output(output_ids)
    print("Decoded Grid:")
    print(tokenizer.grid_to_string(decoded_grid))
    
    # Check vocabulary size
    print(f"Input Vocab Size: {tokenizer.input_vocab_size}")
    print(f"Output Vocab Size: {tokenizer.output_vocab_size}") 