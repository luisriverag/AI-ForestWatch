# Data Flow Analysis: Dataset and DataLoader Classes

## Current Configuration (model_input_size = 128)

### Step-by-Step Data Flow:

#### 1. **Landsat8TrainDataLoader Initialization**
```python
# From config.json:
batch_size = 16
model_input_size = 128
bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # 18 bands
```

#### 2. **Data Splitting Process**
```python
# Full dataset is split into:
# - 80% training data
# - 10% validation data  
# - 10% test data
```

#### 3. **BaseTrainDataset Initialization**
```python
# Key parameters:
stride = 8  # For training mode
model_input_size = 128
bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # Converted to 0-indexed
```

#### 4. **Data Map Generation (First Time)**
```python
# For each pickle file in data_dir:
# - Load full image and label
# - Skip if < 160 valid pixels
# - Generate patches with stride=8
# - Each patch: (example_path, row, col)
```

**Shape Analysis:**
- **Original Image**: `(256, 256, 18)` - Fixed size
- **Patches Generated**: `(256-128)/8 + 1` × `(256-128)/8 + 1` = 17 × 17 = 289 patches per image
- **Each Patch**: `(128, 128, 18)`

#### 5. **__getitem__ Method (Per Sample)**
```python
# Input: k (index)
# Output: (image_tensor, label_tensor)
```

**Step-by-step processing:**

1. **Load Patch Coordinates**: `(example_path, this_row, this_col)`

2. **Load Full Image**: 
   ```python
   example_subset.shape = (H, W, 18)  # Full image
   label_subset.shape = (H, W)        # Full label
   ```

3. **Extract Patch**:
   ```python
   this_example_subset = example_subset[this_row:this_row+128, this_col:this_col+128, :]
   # Shape: (128, 128, 18)
   
   this_label_subset = label_subset[this_row:this_row+128, this_col:this_col+128]
   # Shape: (128, 128)
   ```

4. **Add Spectral Indices** (get_indices function):
   ```python
   # Adds additional channels (NDVI, NDWI, etc.)
   this_example_subset = get_indices(this_example_subset)
   # Shape: (128, 128, 18)  # Same number of channels
   ```

5. **Select Bands**:
   ```python
   this_example_subset = this_example_subset[:, :, self.bands]
   # Shape: (128, 128, 18)  # All 18 bands selected
   ```

6. **Data Augmentation** (training mode only):
   ```python
   # Random horizontal/vertical flips
   # Shape remains: (128, 128, 18)
   ```

7. **Convert to Tensor**:
   ```python
   this_example_subset = transforms.ToTensor()(this_example_subset)
   # Shape: (18, 128, 128)  # Channels first
   
   this_label_subset = transforms.ToTensor()(this_label_subset)
   # Shape: (1, 128, 128)   # Single channel
   ```

#### 6. **DataLoader Batching**
```python
# DataLoader collects batch_size samples
batch_images.shape = (16, 18, 128, 128)  # (batch_size, channels, height, width)
batch_labels.shape = (16, 1, 128, 128)   # (batch_size, 1, height, width)
```

---

## If model_input_size = 512

### What Changes:

#### 1. **Memory Requirements**
```python
# Current (128x128):
# - Per sample: 18 × 128 × 128 = 294,912 values
# - Per batch: 16 × 294,912 = 4,718,592 values

# With 512x512:
# - Per sample: 18 × 512 × 512 = 4,718,592 values (16x larger!)
# - Per batch: 16 × 4,718,592 = 75,497,472 values (16x larger!)
```

#### 2. **Patch Generation**
```python
# Current: stride=8, patches = (256-128)/8 + 1 × (256-128)/8 + 1
# With 512: patches = (256-512)/8 + 1 × (256-512)/8 + 1

# Example: Original image (256, 256)
# Current: (256-128)/8 + 1 = 17 × 17 = 289 patches
# With 512: (256-512)/8 + 1 = NEGATIVE! ❌ IMPOSSIBLE
```

#### 3. **DataLoader Output Shapes**
```python
# Current:
batch_images.shape = (16, 18, 128, 128)
batch_labels.shape = (16, 1, 128, 128)

# With 512:
batch_images.shape = (16, 18, 512, 512)  # 16x larger memory per batch
batch_labels.shape = (16, 1, 512, 512)   # 16x larger memory per batch
```

#### 4. **Training Implications**

**❌ CRITICAL ISSUE:**
- **Original images are only 256×256 pixels**
- **Cannot extract 512×512 patches from 256×256 images**
- **This would cause a runtime error or crash**

**What would happen:**
```python
# In BaseTrainDataset.__init__():
row_limit = label.shape[0] - model_input_size  # 256 - 512 = -256 ❌
col_limit = label.shape[1] - model_input_size  # 256 - 512 = -256 ❌

# This would cause negative limits, leading to:
# - No patches generated
# - Empty dataset
# - Training failure
```

#### 5. **Alternative Solutions**

**Option 1: Use Full 256×256 Images**
```python
# In config.json:
{
    "train_data_loader": {
        "args": {
            "batch_size": 4,  # Reduce from 16 to 4-8
            "model_input_size": 256,  # Use full image size
            # ... other args
        }
    }
}
```

**Option 2: Use Smaller Patches (64×64)**
```python
# In config.json:
{
    "train_data_loader": {
        "args": {
            "batch_size": 32,  # Can increase batch size
            "model_input_size": 64,  # Smaller patches
            # ... other args
        }
    }
}
```

**Option 3: Upscale Images (Not Recommended)**
- Would require preprocessing to resize 256×256 → 512×512
- Loses original resolution quality
- Not recommended for satellite imagery

#### 6. **GPU Memory Estimation**
```python
# Current (128x128, batch_size=16):
# ~2-4 GB GPU memory

# With 256x256, batch_size=4:
# ~4-8 GB GPU memory (manageable)

# With 64x64, batch_size=32:
# ~1-2 GB GPU memory (very efficient)
```

---

## Summary

**❌ 512x512 is IMPOSSIBLE** with 256×256 original images!

**Current (128x128)**: Good balance of context and efficiency
**256x256**: Use full image size, requires batch size reduction
**64x64**: More patches, higher batch size, less context

**Recommendation**: Stick with 128×128 or try 256×256 with reduced batch size.