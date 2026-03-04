import pandas as pd
df = pd.read_json('curated_gcp_marks.json').T
# Unpack the coordinates and drop the nested dictionary
df[['x', 'y']] = df['mark'].apply(pd.Series)
df = df.drop(columns=['mark'])
df = df.reset_index(names='image_path')
initial_count = len(df)
df_clean = df.dropna(subset=['true_width']).copy()

final_count = len(df_clean)
print(f"Dropped {initial_count - final_count} missing/broken rows.")
print(f"Remaining training samples: {final_count}")

# Save this to a new variable so you don't accidentally use the broken paths later
df = df_clean
import pandas as pd
import numpy as np

# --- STEP 0: The NaN & Index Fix ---
# 1. Remove any rows where the shape is missing
df_clean = df_clean.dropna(subset=['verified_shape']).copy()
# 2. Reset index so it's a clean 0 to N-1 range
df_clean = df_clean.reset_index(drop=True)

print(f"Verified {len(df_clean)} samples with valid labels.")

# --- STEP 1: Spatial Outlier Check ---
df_clean = df_clean[
    (df_clean['x'] >= 0) & (df_clean['x'] <= df_clean.get('true_width', 4000)) &
    (df_clean['y'] >= 0) & (df_clean['y'] <= df_clean.get('true_height', 3000))
].copy()

# --- STEP 2: Synthetic L-Shape Flagging ---
df_clean['final_class'] = df_clean['verified_shape'] 
square_indices = df_clean[df_clean['verified_shape'] == 'Square'].index

np.random.seed(42) # For thesis consistency
l_shape_count = int(len(square_indices) * 0.25)
l_indices = np.random.choice(square_indices, size=l_shape_count, replace=False)
df_clean.loc[l_indices, 'final_class'] = 'L-Shaped'

# --- STEP 3: Safe Mapping with Defaults ---
label_map = {'Square': 0, 'Cross': 1, 'L-Shaped': 2}
df_clean['label_idx'] = df_clean['final_class'].map(lambda x: label_map.get(x, 0))

# --- STEP 4: The Weighted Sampling Fix ---
# Using .get() here prevents the KeyError: nan if a value is weird
counts = df_clean['final_class'].value_counts().to_dict()
df_clean['sample_weight'] = df_clean['final_class'].map(lambda x: 1.0 / counts.get(x, 1))
import os
from PIL import Image
import torchvision.transforms.functional as F
import torch
from IPython.display import display # Needed to show images in Notebooks

def extract_reflective_patch_debug(full_image_path, center_x, center_y, patch_size=512):
    # STEP 1: Path Check
    print(f"Checking path: {full_image_path}")
    if not os.path.exists(full_image_path):
        print("❌ ERROR: File not found. Check the path spelling and extension!")
        return None, None
    
    print("✅ File found! Loading image...")
    
    # STEP 2: Load Image
    img = Image.open(full_image_path).convert("RGB")
    print(f"Original Image Size: {img.size}")
    
    # STEP 3: Apply Padding
    pad_val = patch_size // 2
    img_padded = F.pad(img, padding=pad_val, padding_mode='reflect')
    print(f"Padded Image Size: {img_padded.size} (Added {pad_val}px to all sides)")
    
    # STEP 4: Coordinate Shift
    padded_x = center_x + pad_val
    padded_y = center_y + pad_val
    
    left = int(padded_x - pad_val)
    top = int(padded_y - pad_val)
    
    # STEP 5: Crop
    patch = img_padded.crop((left, top, left + patch_size, top + patch_size))
    print(f"Patch extracted successfully at shifted coordinates ({padded_x}, {padded_y})")
    
    local_coords = torch.tensor([0.5, 0.5])
    return patch, local_coords
from PIL import Image

def apply_flexible_sector_mask(patch, center_x, center_y, start_sector=0, num_sectors=2):
    """
    Creates a flexible geometric mask by sweeping through octants.
    - start_sector: 0 to 7 (where the cut begins)
    - num_sectors: how many 45-degree slices to remove (2 = 90-degree 'L')
    """
    patch_np = np.array(patch).copy()
    h, w, _ = patch_np.shape
    
    # 1. Build the polar coordinate map
    y_grid, x_grid = np.ogrid[:h, :w]
    # Calculate angle of every pixel relative to sub-pixel centroid
    angles = (np.arctan2(y_grid - center_y, x_grid - center_x) + 2*np.pi) % (2*np.pi)
    
    # 2. Map angles to 8 sectors (0-7)
    sector_indices = (angles / (np.pi / 4)).astype(int)
    
    # 3. Define the 'mask list' dynamically
    # This handles wrapping (e.g., if start is 7 and num is 2, it masks 7 and 0)
    mask_list = [(start_sector + i) % 8 for i in range(num_sectors)]
    
    # 4. Apply environmental mean color
    bg_color = patch_np.mean(axis=(0, 1)).astype(np.uint8)
    
    for s in mask_list:
        patch_np[sector_indices == s] = bg_color
        
    return Image.fromarray(patch_np)
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import os
import numpy as np

class MarkerDataset(Dataset):
    def __init__(self, metadata_df, base_dir, patch_size=512, transform=None):
        """
        Custom Dataset
        - metadata_df: Balanced dataframe with Square, Cross, and L-Shape labels.
        - base_dir: Root directory of your drone imagery.
        """
        self.df = metadata_df.reset_index(drop=True)
        self.base_dir = base_dir
        self.patch_size = patch_size
        self.pad_val = patch_size // 2 
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.base_dir, row['image_path'])
        
        # 1. Load and Reflective Pad to handle boundary markers
        img = Image.open(img_path).convert("RGB")
        img_padded = F.pad(img, padding=self.pad_val, padding_mode='reflect')

        # 2. Extract 512x512 Patch centered on Global Coordinates
        padded_x = row['x'] + self.pad_val
        padded_y = row['y'] + self.pad_val
        left = int(padded_x - self.pad_val)
        top = int(padded_y - self.pad_val)
        patch = img_padded.crop((left, top, left + self.patch_size, top + self.patch_size))

        # 3. Calculate Local Centroid (Vertex Anchor)
        local_x = padded_x - left
        local_y = padded_y - top

        # 4. Apply RIGOROUS SECTOR MASKING for L-Shapes
        # This physically converts the square into the minority class shape
        if row['final_class'] == 'L-Shape':
            # Randomize start_sector (0-7) for rotational diversity
            start_s = np.random.randint(0, 8)
            # 2 sectors = 90° cut, ensuring < 6 sectors remain occupied
            patch = apply_flexible_sector_mask(patch, local_x, local_y, 
                                             start_sector=start_s, num_sectors=2)

        # 5. Coordinate Normalization [0, 1] for Regression Head
        target_coords = torch.tensor([local_x / self.patch_size, 
                                    local_y / self.patch_size], dtype=torch.float32)
        
        # 6. Label Index for Classification Head
        label = torch.tensor(row['label_idx'], dtype=torch.long)

        # 7. Final Transforms
        if self.transform:
            patch = self.transform(patch)
        else:
            patch = F.to_tensor(patch)

        return patch, {'label': label, 'coords': target_coords}
def analyze_chromatic_balance(patch_pil):
        patch_np = np.array(patch_pil)
        
        # Calculate mean intensity per channel
        r_mean = np.mean(patch_np[:,:,0])
        g_mean = np.mean(patch_np[:,:,1])
        b_mean = np.mean(patch_np[:,:,2])
        
        # Standard deviation across channels (Low = Neutral/Grey, High = Tinted)
        chromatic_std = np.std([r_mean, g_mean, b_mean])
        
        return chromatic_std # If this is low, it's likely a neutral marker board
from torchvision import transforms  
# Update your initialization chunk with this:
train_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
from sklearn.model_selection import train_test_split

# 1. 90-10 Split on your balanced metadata
train_df, val_df = train_test_split(
    df_balanced, 
    test_size=0.10, 
    random_state=42, 
    stratify=df_balanced['label_idx'] # Vital for the 87/13 imbalance
)

# 2. Initialize the Datasets
train_dataset = MarkerDataset(train_df, base_dir="Users/saniarawat/Desktop/Skylark-2/train_dataset", transform="train_transforms")
val_dataset = MarkerDataset(val_df, base_dir="Users/saniarawat/Desktop/Skylark-2/train_dataset", transform="val_transforms")

print(f"Training: {len(train_dataset)} | Validation: {len(val_dataset)}")
import torch
import torch.nn as nn
from torchvision import models  # This fixes the NameError
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import optim
# --- INITIALIZATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MarkerModel(num_classes=3).to(device)

# Loss Functions
criterion_cls = nn.CrossEntropyLoss() # For Square, Cross, L-Shape
criterion_reg = nn.MSELoss()           # For sub-pixel [x, y] precision

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# 1. Calculate weights for each class to handle the 87/13 imbalance
class_counts = train_df['label_idx'].value_counts().to_dict()
# Create a weight for each sample in the training set
weights = 1.0 / torch.tensor([class_counts[i] for i in range(len(class_counts))], dtype=torch.float)
sample_weights = weights[train_df['label_idx'].values]

# 2. Setup the Sampler
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# 3. Initialize the DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
import torch
import torch.nn as nn
from torchvision import models

class MarkerModel(nn.Module):
    def __init__(self, num_classes=3):
        super(MarkerModel, self).__init__()
        # Load pre-trained ResNet-18 backbone
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        in_feats = self.backbone.fc.in_features
        
        # Remove the original classification head
        self.backbone.fc = nn.Identity()
        
        # Head A: Classification (Square vs. Cross vs. L-Shape)
        # Directly addresses the 7-vs-5 sector logic
        self.classifier = nn.Sequential(
            nn.Linear(in_feats, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Head B: Sub-pixel Regression (X, Y Coordinates)
        # Optimized for sub-pixel accuracy in Chennai corridors
        self.regressor = nn.Sequential(
            nn.Linear(in_feats, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid() # Keeps coordinates in range [0, 1]
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        coords = self.regressor(features)
        return logits, coords
# 1. Device selection (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Instantiate Model
model = MarkerModel(num_classes=3).to(device)

# 3. Define Multi-Task Loss Functions
criterion_cls = nn.CrossEntropyLoss() # For geometry classification
criterion_reg = nn.MSELoss()           # For coordinate precision

# 4. Optimizer (Adam is best for balancing multi-task gradients)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print(f"Model initialized on: {device}")
# Pass all our defined components into the loop
train_and_validate(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion_cls=criterion_cls,
    criterion_reg=criterion_reg,
    device=device,
    epochs=15  # 15 epochs is a solid start for transfer learning
)
