import os
import shutil
import random
from PIL import Image, ImageOps

# Define the path to your dataset
source_dir = '../../Data/Images_Complete'
target_dir = '../../Data/Images_Training'


def generate_target_dir_and_augment(pics_per_class=180):
    # Remove the target directory if it exists, then create a new one
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    # Iterate through each class directory in the source directory
    for class_dir in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_dir)
        if os.path.isdir(class_path):
            # Create the corresponding directory in the target directory
            target_class_dir = os.path.join(target_dir, class_dir)
            os.makedirs(target_class_dir)

            # Get all images in the current class directory
            images = [img for img in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, img))]

            # Shuffle the images and select the desired number of images
            random.shuffle(images)
            selected_images = images[:pics_per_class]

            # Copy the selected images to the target directory and perform augmentation
            for img in selected_images:
                src_img_path = os.path.join(class_path, img)
                img_name, img_ext = os.path.splitext(img)

                # Copy the original image
                dest_img_path = os.path.join(target_class_dir, img)
                shutil.copy2(src_img_path, dest_img_path)

                # Perform horizontal flip augmentation
                hor_flipped_img_path = os.path.join(target_class_dir, f"{img_name}_hor_flipped{img_ext}")
                with Image.open(src_img_path) as image:
                    hor_flipped_image = ImageOps.mirror(image)
                    if hor_flipped_image.mode == 'RGBA':
                        hor_flipped_image = hor_flipped_image.convert('RGB')
                    hor_flipped_image.save(hor_flipped_img_path)

                # Perform vertical flip augmentation
                ver_flipped_img_path = os.path.join(target_class_dir, f"{img_name}_ver_flipped{img_ext}")
                with Image.open(src_img_path) as image:
                    ver_flipped_image = ImageOps.flip(image)
                    if ver_flipped_image.mode == 'RGBA':
                        ver_flipped_image = ver_flipped_image.convert('RGB')
                    ver_flipped_image.save(ver_flipped_img_path)

        print("Done with class " + str(class_dir.title()))


def generate_target_dir_variable_min(max_pics_per_class=180):
    # Remove the target directory if it exists, then create a new one
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    # Iterate through each class directory in the source directory
    for class_dir in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_dir)
        if os.path.isdir(class_path):
            # Create the corresponding directory in the target directory
            target_class_dir = os.path.join(target_dir, class_dir)
            os.makedirs(target_class_dir)

            # Get all images in the current class directory
            images = [img for img in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, img))]

            # Shuffle the images and select the desired number of images
            random.shuffle(images)
            selected_images = images[:min(max_pics_per_class, len(images))]

            # Copy the selected images to the target directory
            for img in selected_images:
                src_img_path = os.path.join(class_path, img)
                dest_img_path = os.path.join(target_class_dir, img)
                shutil.copy2(src_img_path, dest_img_path)

            print("Done with class " + str(class_dir.title()))


# Generate the target directory with 180 images per class
generate_target_dir_and_augment(pics_per_class=180)
