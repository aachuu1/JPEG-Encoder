# JPEG & Video Compression Algorithms

Complete implementation of image and video compression algorithms in Python, demonstrating the core principles behind JPEG and modern video codecs.

### 1. **Full JPEG Compression** 
- Block-based DCT (Discrete Cosine Transform) processing (8×8 blocks)
- Standard JPEG quantization matrix implementation
- Quality metrics: MSE (Mean Squared Error) and PSNR (Peak Signal-to-Noise Ratio)
- Complete encode/decode pipeline

### 2. **Color Image Compression**
- RGB to YCbCr color space conversion (JPEG standard)
- Separate channel compression (Y, Cb, Cr)
- YCbCr to RGB reconstruction
- Example using `scipy.misc.face()`

### 3. **Adaptive Compression** 
- Target MSE threshold-based compression
- Binary search optimization for quality factor
- Automatic quality adjustment
- Multiple compression levels comparison

### 4. **Video Compression** 
- **Simple method**: Frame-by-frame independent compression
- **Motion compensation**: I-frames and P-frames implementation
- Temporal redundancy exploitation
- Performance comparison (PSNR improvement ~2-5 dB)

##  Requirements
```bash
pip install numpy matplotlib scipy opencv-python
```

Optional (for YouTube video download):
```bash
pip install yt-dlp
```

##  Usage

### Basic JPEG Compression
```python
from scipy import misc
X = misc.ascent()
X_compressed = jpeg_encode_decode_full_image(X, Q_jpeg)
```

### Color Image Compression
```python
face = misc.face()
face_compressed, _, _ = jpeg_compress_color_image(face.astype(float), Q_jpeg)
```

### Adaptive Compression
```python
result, history = jpeg_compress_with_mse_threshold(X, Q_jpeg, mse_target=100)
```

### Video Compression
```python
# Generated test video
video = create_test_video(frames=30, size=(128, 128))




