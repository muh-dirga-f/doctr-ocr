import time
import os
from doctr.io import DocumentFile
from app import create_doctr_model

def benchmark_create_model():
    start_time = time.time()
    model = create_doctr_model()
    end_time = time.time()
    return end_time - start_time

def benchmark_process_image(model, image_path):
    start_time = time.time()
    doc = DocumentFile.from_images(image_path)
    result = model(doc)
    # print(result)
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    # Benchmark model creation
    model_creation_time = benchmark_create_model()
    print(f"Model creation time: {model_creation_time:.2f} seconds")

    # Create model for image processing benchmark
    model = create_doctr_model()

    # Benchmark image processing
    image_dir = "./image-test"
    total_processing_time = 0
    image_count = 0

    for filename in os.listdir(image_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_dir, filename)
            processing_time = benchmark_process_image(model, image_path)
            total_processing_time += processing_time
            image_count += 1
            print(f"Processing time for {filename}: {processing_time:.2f} seconds")

    if image_count > 0:
        avg_processing_time = total_processing_time / image_count
        print(f"\nAverage processing time per image: {avg_processing_time:.2f} seconds")
    else:
        print("No images found in the specified directory.")
