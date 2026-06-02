from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_rgb_image(path):
    """Load an image as an RGB uint8 numpy array."""
    return np.array(Image.open(path).convert('RGB'), dtype=np.uint8)


# ------------------------------
# Manual grayscale conversion
# ------------------------------
def rgb_to_grayscale(rgb):
    """Convert RGB image to grayscale manually."""
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return np.clip(np.round(gray), 0, 255).astype(np.uint8)


# ------------------------------
# Manual histogram
# ------------------------------
def compute_histogram(channel):
    """Compute 256-bin histogram manually without built-in histogram functions."""
    hist = np.zeros(256, dtype=np.int64)
    flat = channel.flatten()
    for pixel in flat:
        hist[int(pixel)] += 1
    return hist


# ------------------------------
# Manual histogram equalization
# ------------------------------
def equalize_channel(channel):
    """Equalize one uint8 channel manually using histogram + CDF."""
    hist = compute_histogram(channel)
    total_pixels = channel.size

    pdf = hist / total_pixels
    cdf = np.cumsum(pdf)

    # Normalize CDF to 0~255 mapping
    transform = np.round(cdf * 255).astype(np.uint8)
    equalized = transform[channel]
    return equalized, hist, compute_histogram(equalized)


# ------------------------------
# RGB channel-wise equalization
# ------------------------------
def equalize_rgb_separately(rgb):
    """Equalize R, G, B channels independently, then merge."""
    r_eq, _, _ = equalize_channel(rgb[:, :, 0])
    g_eq, _, _ = equalize_channel(rgb[:, :, 1])
    b_eq, _, _ = equalize_channel(rgb[:, :, 2])
    merged = np.stack([r_eq, g_eq, b_eq], axis=2)
    return merged


# ------------------------------
# Brightness-based equalization
# ------------------------------
def equalize_brightness(rgb):
    """Equalize grayscale intensity and rescale RGB while preserving color ratio."""
    rgb_float = rgb.astype(np.float32)
    gray = rgb_to_grayscale(rgb)
    gray_eq, _, _ = equalize_channel(gray)

    ratio = gray_eq.astype(np.float32) / np.maximum(gray.astype(np.float32), 1.0)
    result = rgb_float * ratio[:, :, None]
    return np.clip(np.round(result), 0, 255).astype(np.uint8)


# ------------------------------
# Save helpers
# ------------------------------
def save_image(array, path):
    Image.fromarray(array).save(path)


# ------------------------------
# Plot histogram helper
# ------------------------------
def save_histogram_plot(hist, title, output_path):
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(256), hist, width=1.0)
    plt.title(title)
    plt.xlabel('Intensity Value')
    plt.ylabel('Pixel Count')
    plt.xlim(0, 255)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ------------------------------
# Create montage for one image
# ------------------------------
def save_montage(original_rgb, gray, gray_eq, rgb_eq, brightness_eq, output_path, title):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    panels = [
        (original_rgb, 'Original RGB', False),
        (gray, 'Grayscale', True),
        (gray_eq, 'Grayscale Equalized', True),
        (rgb_eq, 'RGB Channel-wise Equalized', False),
        (brightness_eq, 'Brightness Equalized', False),
    ]

    for ax, (img, name, is_gray) in zip(axes, panels):
        if is_gray:
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            ax.imshow(img)
        ax.set_title(name)
        ax.axis('off')

    axes[-1].axis('off')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


# ------------------------------
# Process one image
# ------------------------------
def process_image(name, image_path, output_dir):
    rgb = load_rgb_image(image_path)

    gray = rgb_to_grayscale(rgb)
    gray_eq, gray_hist_before, gray_hist_after = equalize_channel(gray)
    rgb_eq = equalize_rgb_separately(rgb)
    brightness_eq = equalize_brightness(rgb)

    # Save images
    save_image(rgb, output_dir / f'{name}_original.png')
    save_image(gray, output_dir / f'{name}_gray.png')
    save_image(gray_eq, output_dir / f'{name}_gray_equalized.png')
    save_image(rgb_eq, output_dir / f'{name}_rgb_equalized.png')
    save_image(brightness_eq, output_dir / f'{name}_brightness_equalized.png')

    # Save histograms (grayscale only, per assignment)
    save_histogram_plot(
        gray_hist_before,
        f'{name} - Grayscale Histogram Before Equalization',
        output_dir / f'{name}_gray_hist_before.png'
    )
    save_histogram_plot(
        gray_hist_after,
        f'{name} - Grayscale Histogram After Equalization',
        output_dir / f'{name}_gray_hist_after.png'
    )

    # Save montage
    save_montage(
        rgb,
        gray,
        gray_eq,
        rgb_eq,
        brightness_eq,
        output_dir / f'{name}_montage.png',
        title=name.replace('_', ' ').title(),
    )

    # Simple contrast stats for discussion
    summary = {
        'name': name,
        'gray_std_before': float(np.std(gray)),
        'gray_std_after': float(np.std(gray_eq)),
        'brightness_std_after': float(np.std(rgb_to_grayscale(brightness_eq))),
    }
    return summary


# ------------------------------
# Main program
# ------------------------------
def main():
    # Put your image files in the SAME folder as this .py file,
    # then rename them here if needed.
    images = {
        'image_A_night_sky': 'img1.png',
        'image_B_castle_scene': 'img2.png',
        'image_C_fcu_building': 'img3.png',
    }

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = []

    for name, filename in images.items():
        image_path = script_dir / filename
        if not image_path.exists():
            print(f'[WARNING] File not found: {image_path}')
            print('Please place the image in the same folder as this script, or edit the filename in main().')
            continue

        result = process_image(name, image_path, output_dir)
        summary_lines.append(
            f"{result['name']}: gray std before={result['gray_std_before']:.2f}, "
            f"gray std after={result['gray_std_after']:.2f}, "
            f"brightness-based gray std={result['brightness_std_after']:.2f}"
        )

    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    print(f'Done. Outputs stored in: {output_dir}')
    if summary_lines:
        print('Summary:')
        for line in summary_lines:
            print(line)


if __name__ == '__main__':
    main()
