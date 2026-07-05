import os
import random
from PIL import Image, ImageDraw

def create_pill_image(defective=False):
    # 256x256 image with black background
    img = Image.new('RGB', (256, 256), color=(10, 10, 15))
    draw = ImageDraw.Draw(img)
    
    # Randomize pill color slightly
    r = random.randint(200, 255)
    g = random.randint(200, 255)
    b = random.randint(200, 255)
    
    # Draw central ellipse (pill)
    draw.ellipse([64, 32, 192, 224], fill=(r, g, b))
    
    if defective:
        # Add a "chip" or "hole" (black circle) somewhere on the pill
        cx = random.randint(64, 192)
        cy = random.randint(32, 224)
        radius = random.randint(15, 30)
        draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], fill=(10, 10, 15))
        
        # Maybe add a "scratch" (black line)
        if random.random() > 0.5:
            sx = random.randint(64, 192)
            sy = random.randint(32, 224)
            draw.line([sx, sy, sx+40, sy+40], fill=(10, 10, 15), width=5)
            
    # Add slight random noise to the image
    pixels = img.load()
    for i in range(256):
        for j in range(256):
            if random.random() < 0.05:
                noise = random.randint(-10, 10)
                pr, pg, pb = pixels[i, j]
                pixels[i, j] = (
                    max(0, min(255, pr + noise)),
                    max(0, min(255, pg + noise)),
                    max(0, min(255, pb + noise))
                )
                
    return img

def generate_vision_dataset(base_dir: str):
    train_dir = os.path.join(base_dir, "train", "good")
    test_good_dir = os.path.join(base_dir, "test", "good")
    test_defect_dir = os.path.join(base_dir, "test", "defective")
    
    for d in [train_dir, test_good_dir, test_defect_dir]:
        os.makedirs(d, exist_ok=True)
        
    print(f"Generating training images in {train_dir}...")
    for i in range(50):
        img = create_pill_image(defective=False)
        img.save(os.path.join(train_dir, f"train_{i:03d}.png"))
        
    print(f"Generating test images in {test_good_dir} and {test_defect_dir}...")
    for i in range(10):
        img = create_pill_image(defective=False)
        img.save(os.path.join(test_good_dir, f"test_good_{i:03d}.png"))
        
        img_defect = create_pill_image(defective=True)
        img_defect.save(os.path.join(test_defect_dir, f"test_defect_{i:03d}.png"))

    print("Synthetic vision dataset generated successfully.")

if __name__ == "__main__":
    generate_vision_dataset("data/raw/vision")
