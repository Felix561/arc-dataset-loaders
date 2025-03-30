import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
from config import ARC_V1_DATA_PATH, ARC_V2_DATA_PATH, DatasetConfig, TokenizerConfig, ModelConfig
from arc_dataset import create_arc_v1_dataloader, create_arc_v2_dataloader

class TransformerModel(nn.Module):
    """
    Simple transformer model for demonstrating ARC dataset training.
    """
    def __init__(self, vocab_size=15, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(TokenizerConfig.MAX_SEQ_LENGTH, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=512, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        
    def forward(self, input_ids, attention_mask=None, output_ids=None):
        # Get seq length for positional encodings
        seq_length = input_ids.size(1)
        
        # Create position ids
        pos_ids = torch.arange(0, seq_length, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        
        # Get embeddings and add positional encodings
        x = self.embedding(input_ids) + self.pos_encoder(pos_ids)
        
        # Apply transformer encoder
        if attention_mask is not None:
            # Creating a mask for the transformer from the attention_mask
            transformer_mask = (1 - attention_mask) * -1e9
            transformer_mask = transformer_mask.unsqueeze(1).unsqueeze(2)
            x = self.transformer_encoder(x, src_key_padding_mask=(attention_mask == 0))
        else:
            x = self.transformer_encoder(x)
        
        # For autoregressive behavior, we would set up a causal mask here
        # For simplicity, we'll just use the encoded representation directly
        
        # In a real model, we'd feed the decoder with shifted outputs for training
        # But here we use encoder outputs directly for simplicity
        decoder_output = self.transformer_decoder(x, x)
        
        # Project to vocabulary
        outputs = self.output_layer(decoder_output)
        return outputs


def print_batch_info(batch, device, i):
    """Print useful information about a batch."""
    print(f"\nBatch {i+1} information:")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Output IDs shape: {batch['output_ids'].shape}")
    print(f"  Attention mask shape: {batch['attention_mask'].shape}")
    print(f"  Device: {device}")
    
    # Print a sample input grid in token form
    sample_idx = 0
    tokens = batch['input_ids'][sample_idx].tolist()
    print(f"\n  Sample Input Tokens (ID {sample_idx}):")
    print(f"  {tokens}")
    
    # Print a sample output grid in token form
    tokens = batch['output_ids'][sample_idx].tolist()
    print(f"\n  Sample Output Tokens (ID {sample_idx}):")
    print(f"  {tokens}")
    
    # Print attention mask
    mask = batch['attention_mask'][sample_idx].tolist()
    print(f"\n  Attention Mask (ID {sample_idx}):")
    print(f"  {mask}")


def train_model(version="v1", num_epochs=3, learning_rate=1e-4, print_every=5):
    """
    Train a transformer model on ARC dataset.
    
    Args:
        version: 'v1' or 'v2' for ARC version
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        print_every: Print detailed information every N batches
    """
    # Select the appropriate dataloader based on version
    if version.lower() == "v1":
        data_path = DatasetConfig.ARCV1.DATA_PATH
        create_dataloader = create_arc_v1_dataloader
        version_name = "ARC-V1"
    else:  # v2
        data_path = DatasetConfig.ARCV2.DATA_PATH
        create_dataloader = create_arc_v2_dataloader
        version_name = "ARC-V2"
    
    # Get configuration parameters
    batch_size = ModelConfig.BATCH_SIZE
    shuffle = DatasetConfig.SHUFFLE_DATA
    
    # Create dataloaders - we ensure by_task=False for LLM training
    print(f"\n=== Creating DataLoaders for {version_name} ===")
    train_dataloader = create_dataloader(
        data_path=data_path,
        split="training",
        batch_size=batch_size,
        shuffle=shuffle,
        by_task=False  # Make sure we use LLMDataset for training
    )
    
    val_dataloader = create_dataloader(
        data_path=data_path,
        split="evaluation",
        batch_size=batch_size,
        shuffle=False,
        by_task=False  # Make sure we use LLMDataset for validation
    )
    
    print(f"Training examples: {len(train_dataloader.dataset)}")
    print(f"Validation examples: {len(val_dataloader.dataset)}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = TransformerModel().to(device)
    
    # Print model summary
    print("\n=== Model Architecture ===")
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens (token_id=0)
    
    print(f"\n=== Starting Training ===")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        model.train()
        total_loss = 0
        
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            output_ids = batch["output_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Print detailed batch information every `print_every` batches
            if i % print_every == 0:
                print_batch_info(batch, device, i)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Reshape for CrossEntropyLoss
            # outputs: (batch_size, seq_len, vocab_size)
            # output_ids: (batch_size, seq_len)
            
            # Fix: Ensure outputs and targets have the same length
            outputs = outputs[:, :output_ids.size(1), :]
            
            loss = criterion(
                outputs.contiguous().view(-1, outputs.size(-1)),
                output_ids.view(-1)
            )
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            
            if i % print_every == 0:
                print(f"  Batch {i+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
        
        # Calculate average loss
        avg_train_loss = total_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                # Move data to device
                input_ids = batch["input_ids"].to(device)
                output_ids = batch["output_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Calculate loss
                # Fix: Ensure outputs and targets have the same length
                outputs = outputs[:, :output_ids.size(1), :]
                
                loss = criterion(
                    outputs.contiguous().view(-1, outputs.size(-1)),
                    output_ids.view(-1)
                )
                
                # Track loss
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_dataloader)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
    
    print("\n=== Training Complete ===")
    return model


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a transformer model on ARC dataset")
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"], 
                        help="ARC version (v1 or v2)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--print_every", type=int, default=5,
                        help="Print detailed information every N batches")
    
    args = parser.parse_args()
    
    # Train the model
    model = train_model(
        version=args.version,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        print_every=args.print_every
    )
    
    print("\nNote: The output tokenizer encodes the ground-truth labels (output_ids),")
    print("while the model generates predictions in the same format (without <bos>).")
    print("This allows direct comparison. The encoder input (input_ids) contains")
    print("a <bos> token at the beginning, which is typical for transformer models.") 