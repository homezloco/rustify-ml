"""
Numpy-style image augmentation loop - target for rustify-ml acceleration.
Expected gain: 5-20x via Rust ndarray + rayon parallel iter.
"""


def normalize_pixels(pixels, mean, std):
    """Normalize a flat pixel array: (x - mean) / std per element."""
    result = [0.0] * len(pixels)
    for i in range(len(pixels)):
        result[i] = (pixels[i] - mean) / std
    return result


def apply_gamma(pixels, gamma):
    """Apply gamma correction to each pixel value in [0, 1]."""
    result = [0.0] * len(pixels)
    for i in range(len(pixels)):
        result[i] = pixels[i] ** gamma
    return result


def channel_mean(pixels, channels, channel_idx):
    """Compute mean of a single channel from interleaved RGB data."""
    total = 0.0
    count = 0
    for i in range(channel_idx, len(pixels), channels):
        total += pixels[i]
        count += 1
    return total / count if count > 0 else 0.0


if __name__ == "__main__":
    import time

    pixels = [float(i % 256) / 255.0 for i in range(224 * 224 * 3)]
    start = time.time()
    for _ in range(50):
        normalize_pixels(pixels, 0.485, 0.229)
    elapsed = time.time() - start
    print(f"normalize_pixels 224x224x3: {elapsed:.3f}s for 50 iterations")

    start = time.time()
    for _ in range(50):
        apply_gamma(pixels, 2.2)
    elapsed = time.time() - start
    print(f"apply_gamma 224x224x3: {elapsed:.3f}s for 50 iterations")
