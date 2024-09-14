import time
import os
import pytesseract
from PIL import Image

def benchmark_process_image(image_path):
    start_time = time.time()
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    # print(text)
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    # Benchmark image processing
    image_dir = "./image-test"
    total_processing_time = 0
    image_count = 0

    for filename in os.listdir(image_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_dir, filename)
            processing_time = benchmark_process_image(image_path)
            total_processing_time += processing_time
            image_count += 1
            print(f"Processing time for {filename}: {processing_time:.2f} seconds")

    if image_count > 0:
        avg_processing_time = total_processing_time / image_count
        print(f"\nAverage processing time per image: {avg_processing_time:.2f} seconds")
    else:
        print("No images found in the specified directory.")
