# train.py
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.comcnn import ComCNN
from models.reccnn import RecCNN
from utils.dataset import ImagePatchDataset

# --------- Arguments ---------
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/Set5')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--patch_size', type=int, default=40)
parser.add_argument('--stride', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=1e-4)
args = parser.parse_args()

# --------- Device ---------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# --------- Dataset ---------
dataset = ImagePatchDataset(args.data_dir, patch_size=args.patch_size, stride=args.stride)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# --------- Models ---------
com_cnn = ComCNN().to(device)
rec_cnn = RecCNN().to(device)

# --------- Loss & Optimizers ---------
criterion = nn.MSELoss()
opt_com = torch.optim.Adam(com_cnn.parameters(), lr=args.learning_rate)
opt_rec = torch.optim.Adam(rec_cnn.parameters(), lr=args.learning_rate)

# --------- Training Loop ---------
for epoch in range(args.epochs):
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
    for images in loop:
        images = images.to(device)

        # -------- Forward Pass --------
        compressed = com_cnn(images)

        # Upsample compressed to match target size
        compressed_up = F.interpolate(compressed, size=images.shape[2:], mode='bilinear', align_corners=False)

        rec_out = rec_cnn(compressed_up)

        # -------- Loss --------
        loss_com = criterion(compressed_up, images)
        loss_rec = criterion(rec_out, images)
        loss = loss_com + loss_rec

        # -------- Backward --------
        opt_com.zero_grad()
        opt_rec.zero_grad()
        loss.backward()
        opt_com.step()
        opt_rec.step()

        loop.set_postfix({
            'loss': loss.item(),
            'loss_com': loss_com.item(),
            'loss_rec': loss_rec.item()
        })

# -------- Save Models --------
torch.save(com_cnn.state_dict(), './checkpoints/comcnn.pth')
torch.save(rec_cnn.state_dict(), './checkpoints/reccnn.pth')
print("Training finished and models saved!")