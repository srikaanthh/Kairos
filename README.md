
# DCGAN Text-to-Image Generation

A Deep Convolutional Generative Adversarial Network (DCGAN) implementation for text-to-image generation using GLOVE embeddings and COCO dataset.

## 🎯 Features

- **Text-to-Image Generation**: Generate images from text descriptions using DCGAN architecture
- **GLOVE Embeddings**: 300-dimensional word embeddings projected to 1024-dimensional space
- **COCO Dataset**: Trained on Microsoft COCO dataset with image-caption pairs
- **Comprehensive Evaluation**: FID, IS, Text-Image Matching, and CLIP scores
- **Automatic Checkpointing**: Save models every 10 epochs with resume capability
- **Visualization Tools**: Generate GIFs, loss plots, and evaluation comparisons

## 📋 Requirements

- torch==2.0.0
- torchvision
- numpy==1.21.5
- Pillow==10.0.0
- matplotlib
- imageio
- scipy
- h5py==3.6.0

## 🗂️ Dataset

**Microsoft COCO Dataset**: 
- Training images: `train2014/`
- Validation images: `val2014/`
- Captions: `captions_train2014.json`, `captions_val2014.json`
- **GLOVE Embeddings**: `glove.6B.300d.txt` (300-dimensional word vectors)

## 🏗️ Repository Structure

```
├── models/
│   ├── dcgan_model.py          # Generator and Discriminator architectures
│   ├── char_cnn_rnn_model.py   # Text processing models
│   └── net_modules/            # Network components
├── saved_models/                # Trained model checkpoints
│   ├── generator_final.pth
│   ├── discriminator_final.pth
│   └── checkpoint_epoch_*.pth
├── generated_images/            # Generated images and visualizations
│   ├── output_*.png
│   ├── output_gif_*.gif
│   └── evaluation_*.png
├── utils.py                    # Utility functions
├── data_util.py               # Data processing utilities
├── requirements.txt           # Python dependencies
├── glove.6B.300d.txt         # GLOVE word embeddings
└── DCGAN_Text2Image.ipynb    # Main training notebook
```

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download GLOVE Embeddings**:
   - Place `glove.6B.300d.txt` in the project root

3. **Prepare COCO Dataset**:
   - Download COCO 2014 dataset
   - Update paths in the notebook data loading section

4. **Run Training**:
   ```python
   # Execute cells in DCGAN_Text2Image.ipynb
   # Training will run for 70 epochs with automatic checkpointing
   ```

## 📊 Training Configuration

- **Image Size**: 64x64 pixels
- **Batch Size**: 512
- **Epochs**: 70
- **Learning Rate**: 0.0002
- **Noise Dimension**: 100
- **Embedding Dimension**: 1024 (projected from GLOVE 300d)
- **Optimizer**: Adam (β1=0.5, β2=0.999)

## 📈 Results

### Training Performance
- **Total Training Time**: ~24 hours (70 epochs)
- **Checkpointing**: Every 10 epochs
- **Loss Tracking**: Generator and Discriminator losses monitored

### Evaluation Metrics
- **FID Score**: 322.30 (Fréchet Inception Distance)
- **IS Score**: 0.53 (Inception Score)
- **Text-Image Matching**: -0.15 ± 0.40
- **CLIP Score**: 0.70

### Generated Outputs
- **Sample Images**: 20+ generated images per run
- **Training GIFs**: Animated progress visualization
- **Loss Plots**: Generator and Discriminator loss curves
- **Evaluation Comparisons**: Real vs Generated image grids

## 🔧 Model Architecture

### Generator
- Input: Noise vector (100d) + Text embedding (1024d)
- Output: RGB image (3×64×64)
- Architecture: Transposed convolutions with batch normalization

### Discriminator  
- Input: RGB image (3×64×64) + Text embedding (1024d)
- Output: Binary classification (real/fake)
- Architecture: Convolutional layers with leaky ReLU

### Text Processing
- **GLOVE Embeddings**: 300-dimensional word vectors
- **Projection Layer**: Linear layer (300d → 1024d)
- **Caption Processing**: Average word embeddings for sentence representation

## 📝 Usage

### Generate Images
```python
# Load trained model
generator.load_state_dict(torch.load('saved_models/generator_final.pth'))

# Generate from text
noise = torch.randn(1, 100, 1, 1, device=device)
text_embedding = caption_to_embedding("a red car")
generated_image = generator(noise, text_embedding)
```

### Resume Training
```python
# Load checkpoint
start_epoch, G_losses, D_losses = load_checkpoint(
    'saved_models/checkpoint_epoch_50.pth',
    generator, discriminator, optimizer_G, optimizer_D
)
```

## 🎯 Future Improvements

- **Better Text Encoders**: Implement CLIP or BERT-based text encoders
- **Higher Resolution**: Scale to 128×128 or 256×256 images
- **Improved Architecture**: Consider StyleGAN or other advanced architectures
- **Better Evaluation**: Implement human evaluation and more comprehensive metrics
- **Hyperparameter Tuning**: Optimize learning rates, batch sizes, and architecture

## 📚 References

- [DCGAN Paper](https://arxiv.org/abs/1511.06434)
- [GLOVE Embeddings](https://nlp.stanford.edu/projects/glove/)
- [COCO Dataset](https://cocodataset.org/)
- [Text-to-Image GANs](https://arxiv.org/abs/1612.03242)


