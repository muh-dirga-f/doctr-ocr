from flask import Flask, request, jsonify, render_template, send_from_directory
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from PIL import Image
import io
import time

app = Flask(__name__)

def create_doctr_model():
    model = ocr_predictor(
        det_arch="db_mobilenet_v3_large",
        reco_arch="crnn_mobilenet_v3_small",
        pretrained=True,
        assume_straight_pages=True,
        straighten_pages=False,
        detect_orientation=False,
    )
    return model

def process_image_data(model, image):
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Create DocumentFile from bytes
    doc = DocumentFile.from_images(img_byte_arr)
    
    # Perform OCR
    result = model(doc)
    
    # Convert result to dictionary
    output = []
    for page in result.pages:
        page_dict = {"blocks": []}
        for block in page.blocks:
            block_dict = {"lines": []}
            for line in block.lines:
                line_dict = {"words": []}
                for word in line.words:
                    word_dict = {
                        "value": word.value,
                        "confidence": float(word.confidence),
                        "geometry": word.geometry
                    }
                    line_dict["words"].append(word_dict)
                block_dict["lines"].append(line_dict)
            page_dict["blocks"].append(block_dict)
        output.append(page_dict)
    
    return output

# Load the model when the application starts
start_time = time.time()
model = create_doctr_model()
model_load_time = time.time() - start_time
print(f"Model loaded in {model_load_time:.2f} seconds")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ocr', methods=['POST'])
def ocr():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image = request.files['image']
    img_bytes = image.read()
    
    # Convert bytes to PIL Image
    img = Image.open(io.BytesIO(img_bytes))
    
    # Process the image
    start_time = time.time()
    result = process_image_data(model, img)
    ocr_time = time.time() - start_time
    
    return jsonify({
        'ocr_time': ocr_time,
        'result': result,
    })

# Serve static files
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(debug=True)
