
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def merge_images():
    base_dir = os.path.expanduser('~/ish/results_inference_meanteacher')
    img_paths = [
        os.path.join(base_dir, 'vis_meanteacher.png'), # Case 001
        os.path.join(base_dir, 'vis_case004.png'),
        os.path.join(base_dir, 'vis_case006.png')
    ]
    
    titles = ["Case 001 (Dice 0.93)", "Case 004 (Dice 0.85)", "Case 006 (Dice 0.86)"]
    
    # Create a figure with 3 rows
    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    
    for i, ax in enumerate(axes):
        if os.path.exists(img_paths[i]):
            img = mpimg.imread(img_paths[i])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(titles[i], fontsize=14)
        else:
            print(f"Image not found: {img_paths[i]}")
            ax.text(0.5, 0.5, "Image Not Found", ha='center')
    
    plt.tight_layout()
    output_path = os.path.join(base_dir, 'vis_combined.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved combined image to {output_path}")

if __name__ == "__main__":
    merge_images()
