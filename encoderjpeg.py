import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.fft import dctn, idctn

#requirement 1: complete jpeg algorithm including all blocks

def jpeg_encode_decode_full_image(X, Q_jpeg, block_size=8):

    h, w = X.shape
    X_jpeg = np.zeros_like(X, dtype=float)
    
    #process image in 8x8 blocks
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            #extract current block
            block = X[i:i+block_size, j:j+block_size]
            
            #pad block if it's not complete (edge blocks)
            if block.shape[0] < block_size or block.shape[1] < block_size:
                padded_block = np.zeros((block_size, block_size))
                padded_block[:block.shape[0], :block.shape[1]] = block
                block = padded_block
            
            #encoding: apply discrete cosine transform
            y = dctn(block, norm='ortho')
            
            #quantization using jpeg matrix
            y_quantized = np.round(y / Q_jpeg) * Q_jpeg
            
            #decoding: apply inverse dct
            block_reconstructed = idctn(y_quantized, norm='ortho')
            
            #store reconstructed block in output image
            X_jpeg[i:i+block_size, j:j+block_size] = block_reconstructed[:block.shape[0], :block.shape[1]]
    
    return X_jpeg

#load test image
X = misc.ascent()

#standard jpeg quantization matrix
Q_jpeg = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 28, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])

#apply jpeg compression to entire image
X_jpeg_full = jpeg_encode_decode_full_image(X, Q_jpeg)

#calculate mean squared error and peak signal-to-noise ratio
mse = np.mean((X - X_jpeg_full) ** 2)
psnr = 10 * np.log10(255**2 / mse)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(X, cmap='gray')
plt.title('original image')
plt.axis('off')

plt.subplot(132)
plt.imshow(X_jpeg_full, cmap='gray')
plt.title(f'jpeg compressed image\nmse={mse:.2f}, psnr={psnr:.2f}db')
plt.axis('off')

plt.subplot(133)
plt.imshow(np.abs(X - X_jpeg_full), cmap='hot')
plt.title('difference (error)')
plt.colorbar()
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"requirement 1 - jpeg compression statistics:")
print(f"mse: {mse:.4f}")
print(f"psnr: {psnr:.4f} db")
print(f"image size: {X.shape}")

#requirement 2: extend to color images (rgb -> ycbcr)

def rgb_to_ycbcr(rgb_image):
    """
    convert rgb to ycbcr according to jpeg standard
    """
    #transformation matrix according to jpeg standard
    transform_matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])
    
    #reshape for matrix multiplication
    h, w, c = rgb_image.shape
    rgb_flat = rgb_image.reshape(-1, 3).T
    
    #apply transformation
    ycbcr_flat = transform_matrix @ rgb_flat
    
    #adjust offset for cb and cr channels
    ycbcr_flat[1:] += 128
    
    #reshape back to image
    ycbcr_image = ycbcr_flat.T.reshape(h, w, c)
    
    return ycbcr_image

def ycbcr_to_rgb(ycbcr_image):
    """
    convert ycbcr to rgb
    """
    #inverse transformation matrix
    inverse_transform = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ])
    
    #copy and adjust offset
    ycbcr_adjusted = ycbcr_image.copy()
    ycbcr_adjusted[:, :, 1:] -= 128
    
    #reshape for multiplication
    h, w, c = ycbcr_image.shape
    ycbcr_flat = ycbcr_adjusted.reshape(-1, 3).T
    
    #apply transformation
    rgb_flat = inverse_transform @ ycbcr_flat
    
    #clip to [0, 255] and reshape
    rgb_image = np.clip(rgb_flat.T.reshape(h, w, c), 0, 255)
    
    return rgb_image

def jpeg_compress_color_image(rgb_image, Q_jpeg, block_size=8):
    """
    jpeg compression for color images
    """
    #convert rgb to ycbcr
    ycbcr = rgb_to_ycbcr(rgb_image)
    
    #compress each channel separately
    ycbcr_compressed = np.zeros_like(ycbcr)
    for channel in range(3):
        ycbcr_compressed[:, :, channel] = jpeg_encode_decode_full_image(
            ycbcr[:, :, channel], Q_jpeg, block_size
        )
    
    #convert back to rgb
    rgb_reconstructed = ycbcr_to_rgb(ycbcr_compressed)
    
    return rgb_reconstructed, ycbcr, ycbcr_compressed

#example using scipy.misc.face() (if available)
try:
    face = misc.face()
    
    #apply compression
    face_compressed, face_ycbcr, face_ycbcr_comp = jpeg_compress_color_image(
        face.astype(float), Q_jpeg
    )
    
    #calculate mse for color image
    mse_color = np.mean((face - face_compressed) ** 2)
    psnr_color = 10 * np.log10(255**2 / mse_color)
    
    #visualization
    plt.figure(figsize=(18, 10))
    
    plt.subplot(2, 4, 1)
    plt.imshow(face.astype(np.uint8))
    plt.title('original rgb image')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(face_ycbcr[:,:,0], cmap='gray')
    plt.title('y channel (luminance)')
    plt.axis('off')
    
    plt.subplot(2, 4, 3)
    plt.imshow(face_ycbcr[:,:,1], cmap='gray')
    plt.title('cb channel (blue chrominance)')
    plt.axis('off')
    
    plt.subplot(2, 4, 4)
    plt.imshow(face_ycbcr[:,:,2], cmap='gray')
    plt.title('cr channel (red chrominance)')
    plt.axis('off')
    
    plt.subplot(2, 4, 5)
    plt.imshow(face_compressed.astype(np.uint8))
    plt.title(f'compressed image\npsnr={psnr_color:.2f}db')
    plt.axis('off')
    
    plt.subplot(2, 4, 6)
    plt.imshow(face_ycbcr_comp[:,:,0], cmap='gray')
    plt.title('compressed y channel')
    plt.axis('off')
    
    plt.subplot(2, 4, 7)
    plt.imshow(face_ycbcr_comp[:,:,1], cmap='gray')
    plt.title('compressed cb channel')
    plt.axis('off')
    
    plt.subplot(2, 4, 8)
    plt.imshow(face_ycbcr_comp[:,:,2], cmap='gray')
    plt.title('compressed cr channel')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nrequirement 2 - color image compression:")
    print(f"mse: {mse_color:.4f}")
    print(f"psnr: {psnr_color:.4f} db")
    
except AttributeError:
    print("\nscipy.misc.face() is not available in this version.")

#requirement 3 [6p]: extend compression algorithm until mse threshold

def jpeg_compress_with_mse_threshold(X, Q_base, mse_threshold, block_size=8):
    """
    adaptive jpeg compression until reaching target mse
    """
    quality_factor = 1.0
    best_result = None
    history = []
    
    print(f"\nsearching for quality factor to achieve mse < {mse_threshold}:")
    
    #binary search for optimal quality factor
    min_q, max_q = 0.01, 20.0
    
    #first, check if we can achieve the target at all
    Q_min = np.maximum(Q_base * min_q, 1)
    X_test = jpeg_encode_decode_full_image(X, Q_min, block_size)
    mse_min = np.mean((X - X_test) ** 2)
    
    if mse_min > mse_threshold:
        #even if the best quality doesn't meet threshold, return best effort
        print(f"  warning: minimum achievable mse is {mse_min:.2f}, using best quality")
        psnr_min = 10 * np.log10(255**2 / mse_min)
        best_result = {
            'image': X_test,
            'quality': min_q,
            'mse': mse_min,
            'psnr': psnr_min,
            'Q_matrix': Q_min
        }
        return best_result, history
    
    #binary search
    while max_q - min_q > 0.01:
        quality_factor = (min_q + max_q) / 2
        Q_scaled = np.maximum(Q_base * quality_factor, 1)
        
        #compress with current quality setting
        X_compressed = jpeg_encode_decode_full_image(X, Q_scaled, block_size)
        mse = np.mean((X - X_compressed) ** 2)
        psnr = 10 * np.log10(255**2 / mse)
        
        #estimate compression ratio (simplified)
        compression_ratio = quality_factor * 10
        
        history.append({
            'quality': quality_factor,
            'mse': mse,
            'psnr': psnr,
            'compression_ratio': compression_ratio
        })
        
        print(f"  q={quality_factor:.3f}: mse={mse:.2f}, psnr={psnr:.2f}db")
        
        if mse < mse_threshold:
            best_result = {
                'image': X_compressed,
                'quality': quality_factor,
                'mse': mse,
                'psnr': psnr,
                'Q_matrix': Q_scaled
            }
            max_q = quality_factor  #try lower quality (higher compression)
        else:
            min_q = quality_factor  #need higher quality
    
    #if still no result, use the last computed one
    if best_result is None and len(history) > 0:
        #use minimum quality factor that was tested
        quality_factor = min_q
        Q_scaled = np.maximum(Q_base * quality_factor, 1)
        X_compressed = jpeg_encode_decode_full_image(X, Q_scaled, block_size)
        mse = np.mean((X - X_compressed) ** 2)
        psnr = 10 * np.log10(255**2 / mse)
        best_result = {
            'image': X_compressed,
            'quality': quality_factor,
            'mse': mse,
            'psnr': psnr,
            'Q_matrix': Q_scaled
        }
    
    return best_result, history

#test with different mse thresholds
mse_thresholds = [50, 100, 200]

plt.figure(figsize=(18, 6))

for idx, mse_target in enumerate(mse_thresholds):
    result, history = jpeg_compress_with_mse_threshold(X, Q_jpeg, mse_target)
    
    if result is not None:
        plt.subplot(1, 3, idx+1)
        plt.imshow(result['image'], cmap='gray')
        plt.title(f'mse target={mse_target}\n'
                  f'actual mse={result["mse"]:.2f}\n'
                  f'psnr={result["psnr"]:.2f}db\n'
                  f'q factor={result["quality"]:.3f}')
        plt.axis('off')
    else:
        print(f"warning: could not find solution for mse target {mse_target}")

plt.tight_layout()
plt.show()

#requirement 4: extend to video compression

def create_test_video(frames=30, size=(128, 128)):
    """
    create simple test video with moving ball
    """
    video = []
    
    for frame_idx in range(frames):
        #create frame
        img = np.zeros(size)
        
        #ball position follows circular path
        center_x = int(size[0] / 2 + 40 * np.sin(2 * np.pi * frame_idx / frames))
        center_y = int(size[1] / 2 + 40 * np.cos(2 * np.pi * frame_idx / frames))
        
        #draw ball (circle)
        y, x = np.ogrid[:size[0], :size[1]]
        mask = (x - center_x)**2 + (y - center_y)**2 <= 15**2
        img[mask] = 255
        
        video.append(img)
    
    return np.array(video)

def compress_video_as_images(video, Q_jpeg, block_size=8):
    """
    video compression by treating each frame as independent image
    """
    compressed_video = []
    
    for frame in video:
        compressed_frame = jpeg_encode_decode_full_image(frame, Q_jpeg, block_size)
        compressed_video.append(compressed_frame)
    
    return np.array(compressed_video)

def compress_video_with_motion(video, Q_jpeg, block_size=8):
    """
    video compression with motion compensation (simplified)
    uses i-frames and p-frames: first frame is i-frame, rest are p-frames (differences)
    """
    compressed_video = []
    
    #first frame = i-frame (fully compressed)
    i_frame = jpeg_encode_decode_full_image(video[0], Q_jpeg, block_size)
    compressed_video.append(i_frame)
    previous_frame = i_frame
    
    #remaining frames = p-frames (differences from previous frame)
    for frame in video[1:]:
        #calculate difference from previous frame
        diff = frame - previous_frame
        
        #compress difference with higher quality (lower q factor)
        diff_compressed = jpeg_encode_decode_full_image(diff, Q_jpeg * 0.5, block_size)
        
        #reconstruct frame by adding compressed difference
        reconstructed_frame = previous_frame + diff_compressed
        reconstructed_frame = np.clip(reconstructed_frame, 0, 255)
        
        compressed_video.append(reconstructed_frame)
        previous_frame = reconstructed_frame
    
    return np.array(compressed_video)

#generate test video
video = create_test_video(frames=30, size=(128, 128))

#compression without motion compensation
video_compressed_simple = compress_video_as_images(video, Q_jpeg)

#compression with motion compensation
video_compressed_motion = compress_video_with_motion(video, Q_jpeg)

#calculate mse for both methods
mse_simple = np.mean((video - video_compressed_simple) ** 2)
mse_motion = np.mean((video - video_compressed_motion) ** 2)

psnr_simple = 10 * np.log10(255**2 / mse_simple)
psnr_motion = 10 * np.log10(255**2 / mse_motion)

print(f"\nrequirement 4 - video compression:")
print(f"simple compression (no motion compensation):")
print(f"  mse: {mse_simple:.4f}, psnr: {psnr_simple:.4f} db")
print(f"compression with motion compensation:")
print(f"  mse: {mse_motion:.4f}, psnr: {psnr_motion:.4f} db")
print(f"  improvement: {psnr_motion - psnr_simple:.2f} db")

#visualize selected frames
frames_to_show = [0, 10, 20, 29]

plt.figure(figsize=(18, 10))

for idx, frame_idx in enumerate(frames_to_show):
    #original frame
    plt.subplot(3, 4, idx+1)
    plt.imshow(video[frame_idx], cmap='gray', vmin=0, vmax=255)
    plt.title(f'original - frame {frame_idx}')
    plt.axis('off')
    
    #simple compression
    plt.subplot(3, 4, idx+5)
    plt.imshow(video_compressed_simple[frame_idx], cmap='gray', vmin=0, vmax=255)
    plt.title(f'simple compression')
    plt.axis('off')
    
    #motion compensation
    plt.subplot(3, 4, idx+9)
    plt.imshow(video_compressed_motion[frame_idx], cmap='gray', vmin=0, vmax=255)
    plt.title(f'motion compensation')
    plt.axis('off')

plt.tight_layout()
plt.show()
