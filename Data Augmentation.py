import albumentations as A
import cv2
import os
from glob import glob

# Define transformations with increased intensities
transformations = {
    # 'random_rotate_90': A.RandomRotate90(p=1.0),
    # 'perspective': A.Perspective(scale=(0.05, 0.15), p=1.0),
    # 'affine': A.Affine(scale=(0.5, 1.5), rotate=(-60, 60), p=1.0),
    # 'gauss_noise': A.GaussNoise(var_limit=(30, 100), p=1.0),
    # 'random_brightness_contrast': A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
    # 'blur': A.Blur(blur_limit=7, p=1.0),
    # 'optical_distortion': A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1.0),
    # 'grid_distortion': A.GridDistortion(num_steps=10, distort_limit=0.5, p=1.0),
    'hue_saturation_value': A.HueSaturationValue(
        hue_shift_limit=(-10, 10),  # Focus on green tones
        sat_shift_limit=(-30, 30),  # Control saturation to avoid extremes
        val_shift_limit=(-20, 20),  # Control value to maintain brightness
        p=1.0)
}

# Define source and destination directories
source_dir = 'dataset\Original_dataset\Healthy Rice'
destination_dir = 'dataset\Augmented_dataset\Healthy Rice_Augmented'

# Create destination directory if it does not exist
os.makedirs(destination_dir, exist_ok=True)

# Counter for the number of images saved
image_count = 0
max_images = 3000

# Process each image
for image_path in glob(os.path.join(source_dir, '*')):
    if image_count >= max_images:
        print('Reached the maximum number of images. Exiting...')
        break

    image = cv2.imread(image_path)
    if image is None:
        print(f'Failed to load image {image_path}')
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply each transformation and save the result
    for name, transform in transformations.items():
        if image_count >= max_images:
            break

        augmented_image = transform(image=image_rgb)['image']
        
        # Convert back to BGR for saving with OpenCV
        augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
        
        # Create a directory for each transformation
        transform_dir = os.path.join(destination_dir, name)
        os.makedirs(transform_dir, exist_ok=True)

        # Save the transformed image
        filename = os.path.basename(image_path)
        save_path = os.path.join(transform_dir, f'{os.path.splitext(filename)[0]}_{name}_transformed.jpg')
        cv2.imwrite(save_path, augmented_image_bgr)
        print(f'Saved {name} transformed image to {save_path}')

        image_count += 1
        if image_count >= max_images:
            break

print('Processing completed.')
