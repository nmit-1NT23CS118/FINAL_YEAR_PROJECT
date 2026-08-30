import os
import random
from PIL import Image
from torchvision import transforms

def augment_dataset(data_dir, target_count=3000):
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} directory not found.")
        return

    # Define random augmentations
    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])

    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Starting augmentation in {data_dir}. Target: {target_count} images per class.")

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        # List all image files
        files = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        current_count = len(files)
        
        print(f"\nClass: '{cls}'")
        print(f"  Current image count: {current_count}")
        
        if current_count >= target_count:
            print(f"  Already has {current_count} images. No augmentation needed.")
            continue
            
        needed = target_count - current_count
        print(f"  Augmenting {needed} images...")
        
        for i in range(needed):
            orig_file = random.choice(files)
            orig_path = os.path.join(cls_dir, orig_file)
            
            try:
                # Open image
                with Image.open(orig_path) as img:
                    img_rgb = img.convert("RGB")
                    # Apply augmentation
                    aug_img = augment_transform(img_rgb)
                    
                    # Generate a unique new filename
                    orig_name_only, ext = os.path.splitext(orig_file)
                    new_filename = f"aug_{i:04d}_{orig_name_only}{ext}"
                    new_path = os.path.join(cls_dir, new_filename)
                    
                    # Save the augmented image
                    aug_img.save(new_path)
            except Exception as e:
                print(f"  Error augmenting file {orig_file}: {e}")
                
        # Verify the new count
        new_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  Finished class '{cls}'. New count: {len(new_files)}")

if __name__ == "__main__":
    augment_dataset("Data_set/train", 3000)
