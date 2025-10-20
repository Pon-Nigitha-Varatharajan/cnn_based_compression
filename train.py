import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models.comcnn import ComCNN
from models.reccnn import RecCNN
from utils.dataset import ImagePatchDataset

def main():
    parser = argparse.ArgumentParser(description='Train CNN Compression Models')
    parser.add_argument('--data_dir', type=str, default='./data/Set5', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--patch_size', type=int, default=40, help='Patch size')
    parser.add_argument('--stride', type=int, default=20, help='Stride for patch extraction')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_interval', type=int, default=10, help='Save interval in epochs')
    args = parser.parse_args()

    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('./checkpoints', exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = ImagePatchDataset(args.data_dir, patch_size=args.patch_size, stride=args.stride)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Training on {len(dataset)} patches from {args.data_dir}")
    
    # Initialize models
    com_cnn = ComCNN().to(device)
    rec_cnn = RecCNN().to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        list(com_cnn.parameters()) + list(rec_cnn.parameters()), 
        lr=args.learning_rate
    )
    
    print("Starting training...")
    
    for epoch in range(args.epochs):
        com_cnn.train()
        rec_cnn.train()
        
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch_idx, images in enumerate(progress_bar):
            images = images.to(device)
            
            # Forward pass
            compressed = com_cnn(images)
            compressed_up = F.interpolate(compressed, size=images.shape[2:], mode='bilinear', align_corners=False)
            reconstructed = rec_cnn(compressed_up)
            
            # Calculate loss
            loss = criterion(reconstructed, images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.6f}')
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            torch.save(com_cnn.state_dict(), f'./checkpoints/comcnn_epoch_{epoch+1}.pth')
            torch.save(rec_cnn.state_dict(), f'./checkpoints/reccnn_epoch_{epoch+1}.pth')
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    # Save final models
    torch.save(com_cnn.state_dict(), './checkpoints/comcnn.pth')
    torch.save(rec_cnn.state_dict(), './checkpoints/reccnn.pth')
    print("Training completed! Final models saved to checkpoints/")

if __name__ == "__main__":
    main()