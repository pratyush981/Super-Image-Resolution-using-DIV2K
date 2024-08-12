from PIL import Image
import os

# Define path (atleast do this)
input_image_path = 'C:/Users/praty.PRATPC/PycharmProjects/SUPERImageRESOLUTION/.venv/DIV2K_train_HR/0001.png'
output_image_path = 'D:/output/optimized_image.png'

target_size = (1920, 1080)
quality = 95


def optimize_image(input_path, output_path, target_size=(1920, 1080), quality=85):
    try:
        with Image.open(input_path) as img:
            img_resized = img.resize(target_size, Image.LANCZOS)

            format = 'PNG' if output_path.lower().endswith('.png') else 'JPEG'

            img_resized.save(output_path, format=format, quality=quality, optimize=True)

        print(f"Image successfully optimized and saved to {output_path}")
    except Exception as e:
        print(f"Error optimizing image: {e}")

output_dir = os.path.dirname(output_image_path)
os.makedirs(output_dir, exist_ok=True)

optimize_image(input_image_path, output_image_path, target_size, quality)

#You need to have Pillow library installed in your Python environment to run this script.

