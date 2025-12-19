import cv2
import numpy as np
from flask import Flask, request, render_template, session
import logging
from io import BytesIO
import base64
from PIL import Image
import os
import secrets
import re
import json
import uuid  # Import the uuid module

app = Flask(__name__)
logging.info("Flask app initialized")  # ADDED LOGGING
app.secret_key = secrets.token_hex(24)
app.config['SESSION_TYPE'] = 'filesystem'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for default parameter values
DEFAULT_SCALE_FACTOR = 1.05
DEFAULT_MIN_NEIGHBORS = 6
DEFAULT_MIN_SIZE = 30
DEFAULT_SIMILARITY_THRESHOLD = 0.6

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# HTML template (Corrected version with escaped curly braces)
HTML_TEMPLATE_START = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Artificial Intelligence Facial Similarity Analyzer</title>
    <style>
        body {{
            font-family: sans-serif;
            margin: 20px;
            line-height: 1.6;
        }}

        #comparison-matrix {{
            display: grid;
            grid-template-columns: auto repeat(auto-fit, minmax(400px, 1fr)); /* Columns: Labels + Surveillance Images */
            gap: 10px;
            max-width: 1500px;
            margin: 20px auto;
        }}
        /* ADDED CSS */
         #comparison-matrix > div:nth-child(odd) {{
             background-color: #f9f9f9; /* light gray */
         }}
        /* ADDED CSS */

        .matrix-cell {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: center;
        }}

        .face-comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr; /* Three columns: Face 1, Face 2, Heatmap */
            gap: 5px;
        }}

        .face-comparison-grid img {{
            max-width: 100%;
            max-height: 100px;
            display: block;
            margin: 0 auto;
            border: 1px solid #eee;
        }}
         /* ADDED CSS */
        .face-comparison-grid div {{ /* Targets the individual comparison divs */
            border: 1px solid #ddd; /* Subtle border */
            padding: 5px;
            margin-bottom: 5px; /* Add a bit of space between comparisons */
        }}
        /* ADDED CSS */

        .matrix-cell img {{
            max-width: 100%;
            max-height: 150px;
            display: block;
            margin: 0 auto;
            border: 1px solid #eee;
        }}

        .matrix-header {{
            font-weight: bold;
            background-color: #f0f0f0;
        }}

        #explanation-box {{
            max-width: 1200px;
            margin: 20px auto;
            border: 1px solid #ccc;
            padding: 15px;
        }}

        .ai-title {{
            font-size: 2em;
            font-weight: bold;
            color: #000;
            background-color: #eee;
            text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.3);
            text-align: center;
        }}

        h2 {{
            text-align: center;
        }}

        #upload-area {{
            max-width: 1200px;
            margin: 20px auto;
            text-align: center;
        }}

        #parameter-area {{
            max-width: 1200px;
            margin: 20px auto;
            text-align: center;
            border: 1px solid #ccc;
            padding: 15px;
        }}
         /* Style for the face detection image container */
        .face-detection-container {{
            position: relative; /* Make the container relative */
            display: inline-block; /* Adjust to content */
        }}

        /* Style for the remove face button */
        .remove-face-button {{
            position: absolute; /* Position it absolutely within the container */
            background-color: rgba(255, 0, 0, 0.5); /* Semi-transparent red */
            color: white;
            border: none;
            cursor: pointer;
            padding: 2px 5px;
            font-size: 0.8em;
        }}

        /* Highlight on hover */
        .face-detection-container:hover {{
            border: 2px solid blue; /* Highlight the container */
        }}

        /* Visual link on checkbox hover */
        .face-detection-container input[type="checkbox"]:hover + label {{
            background-color: rgba(0, 255, 0, 0.5); /* Slightly darker green on hover */
        }}

        /* Style for the original images with bounding boxes */
        .original-image-container {{
            position: relative;
            display: inline-block;
        }}

        .bounding-box {{
            position: absolute;
            border: 2px solid red;
        }}

    </style>
    <script>
        function removeFace(faceId) {{
            // Call removeSelectedFaces with a single face ID
            removeSelectedFaces([faceId]);
        }}

        function removeSelectedFaces(faceIds) {{
            // If faceIds is not provided, get the selected face IDs from checkboxes
            if (!faceIds) {{
                faceIds = Array.from(document.querySelectorAll('input[name="selected_faces"]:checked'))
                    .map(checkbox => checkbox.value);
            }}

            if (faceIds.length === 0) {{
                alert('No faces selected for removal.');
                return;
            }}

            const formData = new FormData();
            formData.append('selected_face_ids', JSON.stringify(faceIds));
            formData.append('scaleFactor', document.getElementById('scaleFactor').value);
            formData.append('minNeighbors', document.getElementById('minNeighbors').value);
            formData.append('minSize', document.getElementById('minSize').value);
            formData.append('similarityThreshold', document.getElementById('similarityThreshold').value);

            fetch('/remove_faces', {{
                method: 'POST',
                body: formData,
            }})
            .then(response => response.text())
            .then(updatedMatrixHtml => {{
                document.getElementById('comparison-matrix').outerHTML = updatedMatrixHtml; // Replace the comparison-matrix
            }})
            .catch(error => console.error('Error:', error));
        }}
    </script>
</head>
<body>
    <h1 class="ai-title">Artificial Intelligence Facial Similarity Analyzer - Happy 13th Birthday Teddy!</h1>

    <div id="upload-area">
        <h2>Upload Images</h2>
        <form method="POST" enctype="multipart/form-data" action="/" enctype="multipart/form-data">
            <label for="referenceImageInput">Reference Images (Rows):</label>
            <input type="file" id="referenceImageInput" name="referenceImage" multiple><br><br>

            <label for="surveillanceImageInput">Surveillance Images (Columns):</label>
            <input type="file" id="surveillanceImageInput" name="surveillanceImage" multiple><br><br>

            <div id="parameter-area">
                <h2>Adjust Parameters</h2>
                <label for="scaleFactor">Scale Factor:</label>
                <input type="number" id="scaleFactor" name="scaleFactor" value="{scale_factor}" step="0.01"><br><br>

                <label for="minNeighbors">Min Neighbors:</label>
                <input type="number" id="minNeighbors" name="minNeighbors" value="{min_neighbors}" step="1"><br><br>

                <label for="minSize">Min Size:</label>
                <input type="number" id="minSize" name="minSize" value="{min_size}" step="1"><br><br>

                <label for="similarityThreshold">Similarity Threshold:</label>
                <input type="number" id="similarityThreshold" name="similarityThreshold" value="{similarity_threshold}" step="0.01"><br><br>
            </div>

            <button type="submit">Upload and Compare</button>
        </form>
    </div>

    <div id="explanation-box">
        <h2>How This Works</h2>
        <p>This application compares face images using the following steps:</p>
        <ul>
            <li><strong>Face Detection:</strong> Detects faces in each image using a Haar Cascade Classifier.</li>
            <li>Cropped face regions are converted to grayscale.</li>
            <li>Generates a histogram representing the distribution of grayscale values.</li>
            <li>Calculates a similarity score (correlation) by comparing the histograms.</li>
        </ul>
        <p>A higher score indicates greater similarity.</p>
        <p><strong>AI vs. Traditional:</strong> Uses AI for face detection and traditional image processing for comparison.</p>

        <h2>Parameter Tuning Guide</h2>
        <p>Adjust the following parameters to improve face detection accuracy:</p>
        <ul>
            <li><strong>Scale Factor:</strong>
                <ul>
                    <li>Lower values (e.g., 1.02) are more sensitive but can cause more false positives.</li>
                    <li>Higher values (e.g., 1.06) are faster but might miss smaller faces.</li>
                    <li>*Missing Faces:* Decrease Scale Factor (e.g., 1.04 to 1.03 or 1.02)</li>
                </ul>
            </li>
            <li><strong>Min Neighbors:</strong>
                <ul>
                    <li>Specifies the minimum number of neighboring rectangles that must be detected to consider it a face.</li>
                    <li>Higher values reduce false positives.</li>
                    <li>*Detecting Non-Face Objects:* Increase Min Neighbors (e.g., 6 to 8)</li>
                </ul>
            </li>
            <li><strong>Min Size:</strong>
                <ul>
                    <li>Specifies the minimum size (in pixels) of a face.</li>
                    <li>Increase to eliminate very small false positives.</li>
                </ul>
            </li>
        </ul>

        <h3>Troubleshooting Tips:</h3>
        <ul>
            <li><strong>False Positives (Detecting Non-Face Objects):</strong>
                <ol>
                    <li>Increase <strong>Min Neighbors</strong> first.</li>
                    <li>If that doesn't work, slightly increase <strong>Scale Factor</strong>.</li>
                    <li>If detecting very small objects, increase <strong>Min Size</strong>.</li>
                </ol>
            </li>
            <li><strong>False Negatives (Missing Faces):</strong>
                <ol>
                    <li>Decrease <strong>Scale Factor</strong>. This is the most common solution.</li>
                    <li>If decreasing Scale Factor causes too many false positives, try decreasing Min Neighbors slightly.</li>
                </ol>
            </li>
        </ul>
        <p>Iteratively adjust these parameters and test on a variety of images to find the optimal settings for your specific use case.</p>
    </div>
    <button type="button" onclick="removeSelectedFaces()">Remove Selected Faces</button>
"""

HTML_TEMPLATE_END = """
</body>
</html>
"""

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def detect_faces(image, scale_factor, min_neighbors, min_size):
    """Detects faces in an image using the Haar cascade classifier."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                                          minSize=(min_size, min_size))
    return faces


def crop_face(image, face):
    """Crops a face from an image."""
    x, y, w, h = face
    return image[y:y + h, x:x + w]


def calculate_histogram(face_image):
    """Calculates the histogram of a face image."""
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
    hist /= hist.sum()  # Normalize the histogram
    return hist


def compare_histograms(hist1, hist2):
    """Compares two histograms using the correlation method."""
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity


def convert_image_to_base64(image):
    """Converts a OpenCV image to base64 for embedding in HTML."""
    _, buffer = cv2.imencode('.jpg', image)
    io_buf = BytesIO(buffer)
    img_str = base64.b64encode(io_buf.read()).decode('utf-8')
    return img_str


def draw_bounding_boxes(image, faces):
    """Draws bounding boxes on an image."""
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bounding box
    return image


def create_heatmap_image(similarity_score):
    """
    Generates a heatmap image based on the similarity score.

    Args:
        similarity_score (float): The similarity score between two faces (0 to 1).

    Returns:
        str: Base64 encoded string of the heatmap image.
    """

    # Define a colormap (e.g., from red to green)
    # You can experiment with different colormaps (cv2.COLORMAP_JET, cv2.COLORMAP_VIRIDIS, etc.)
    colormap = cv2.COLORMAP_JET

    # Convert the similarity score to a value between 0 and 255
    color_index = int(similarity_score * 255)

    # Create a 1x1 image with the color index
    heatmap_img = np.zeros((1, 1, 1), dtype=np.uint8)
    heatmap_img[0, 0] = color_index

    # Apply the colormap
    heatmap_img = cv2.applyColorMap(heatmap_img, colormap)

    # Convert the heatmap image to base64
    heatmap_base64 = convert_image_to_base64(heatmap_img)

    return heatmap_base64


def fix_image_orientation(image_path):  # Changed to accept image path
    """Fixes the orientation of an image loaded from file data."""
    try:
        image = Image.open(image_path)  # Open image from path
        # Check for EXIF data and orientation tag
        if hasattr(image, '_getexif'):
            exif = image._getexif()
            if exif:
                orientation = exif.get(274)
                if orientation:
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(-90, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
        # Convert to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image
    except Exception as e:
        logging.error(f"Error fixing image orientation: {e}")
        return None


def create_comparison_matrix(reference_image_paths, surveillance_image_paths, scale_factor, min_neighbors, min_size,
                             similarity_threshold, removed_face_ids):
    """
    Generates the HTML for the face comparison matrix.
    """
    reference_images = []
    surveillance_images = []
    reference_face_locations = []  # Store face locations
    surveillance_face_locations = []  # Store face locations

    # Load reference images and detect faces
    reference_faces_data = []
    for image_path in reference_image_paths:
        image = fix_image_orientation(image_path)  # Pass the image path
        if image is None:
            logging.error(f"Failed to load reference image: {image_path}")
            continue
        faces = detect_faces(image, scale_factor, min_neighbors, min_size)
        image_with_boxes = draw_bounding_boxes(image.copy(), faces)  # Draw bounding boxes
        reference_images.append(image_with_boxes)
        reference_faces_data.append({"filename": os.path.basename(image_path), "faces": faces})
        reference_face_locations.append(faces)

    # Load surveillance images and detect faces
    surveillance_faces_data = []
    for image_path in surveillance_image_paths:
        image = fix_image_orientation(image_path)  # Pass the image path
        if image is None:
            logging.error(f"Failed to load surveillance image: {image_path}")
            continue
        faces = detect_faces(image, scale_factor, min_neighbors, min_size)
        image_with_boxes = draw_bounding_boxes(image.copy(), faces)  # Draw bounding boxes
        surveillance_images.append(image_with_boxes)
        surveillance_faces_data.append({"filename": os.path.basename(image_path), "faces": faces})
        surveillance_face_locations.append(faces)

    # Build the HTML table
    html = '<div id="comparison-matrix">'
    html += '<div class="matrix-cell matrix-header"></div>'  # Empty top-left cell

    # Add surveillance image filenames as headers
    for sur_idx, sur_data in enumerate(surveillance_faces_data):
        filename = sur_data["filename"]
        # Add original surveillance image with bounding boxes
        html += f'<div class="matrix-cell matrix-header">'
        html += f'<div class="original-image-container">'
        sur_image = surveillance_images[sur_idx]
        sur_image_base64 = convert_image_to_base64(sur_image)
        html += f'<img src="data:image/jpeg;base64,{sur_image_base64}" alt="Original Surveillance Image with Bounding Boxes"/>'
        html += f'<p>{filename}</p>'  # Filename below the image
        html += f'</div>'  # Close original-image-container
        html += f'</div>'  # Close matrix-cell matrix-header

    # Iterate through reference images
    for ref_idx, ref_data in enumerate(reference_faces_data):
        ref_filename = ref_data["filename"]
        ref_image = reference_images[ref_idx]
        ref_faces = ref_data["faces"]

        # Add reference image filename as header
        html += f'<div class="matrix-cell matrix-header">'
        html += f'<div class="original-image-container">'
        ref_image_base64 = convert_image_to_base64(ref_image)
        html += f'<img src="data:image/jpeg;base64,{ref_image_base64}" alt="Original Reference Image with Bounding Boxes"/>'
        html += f'<p>{ref_filename}</p>'  # Filename below the image
        html += f'</div>'  # Close original image container
        html += '</div>'  # Close matrix-cell matrix-header

        # Iterate through surveillance images
        for sur_idx, sur_data in enumerate(surveillance_faces_data):
            sur_filename = sur_data["filename"]
            sur_image = surveillance_images[sur_idx]
            sur_faces = sur_data["faces"]

            # Create a grid for face comparisons and heatmap
            html += '<div class="matrix-cell">'
            html += '<div class="face-comparison-grid">'

            # Iterate through faces in the reference image
            for ref_face_idx, ref_face in enumerate(ref_faces):
                ref_face_id = f'ref_{ref_idx}_{ref_face_idx}'
                ref_face_cropped = crop_face(ref_image, ref_face)
                ref_hist = calculate_histogram(ref_face_cropped)
                ref_face_base64 = convert_image_to_base64(ref_face_cropped)

                # Iterate through faces in the surveillance image
                for sur_face_idx, sur_face in enumerate(sur_faces):
                    sur_face_id = f'sur_{sur_idx}_{sur_face_idx}'
                    comparison_id = f'{ref_face_id}_vs_{sur_face_id}'

                    # Check if this face comparison has been removed
                    if comparison_id in removed_face_ids:
                        continue

                    sur_face_cropped = crop_face(sur_image, sur_face)
                    sur_hist = calculate_histogram(sur_face_cropped)
                    sur_face_base64 = convert_image_to_base64(sur_face_cropped)

                    similarity_score = compare_histograms(ref_hist, sur_hist)

                    # Generate heatmap image
                    heatmap_base64 = create_heatmap_image(similarity_score)

                    # Show only faces with similarity above the threshold
                    if similarity_score >= similarity_threshold:
                        # Create a unique identifier for the face detection container
                        container_id = f"face-container-{ref_idx}-{ref_face_idx}-{sur_idx}-{sur_face_idx}"

                        # Add face images and heatmap to the grid with the remove button
                        html += f'<div id="{container_id}" class="face-detection-container">'
                        html += f'<img src="data:image/jpeg;base64,{ref_face_base64}" alt="Reference Face"/>'
                        html += f'<img src="data:image/jpeg;base64,{sur_face_base64}" alt="Surveillance Face"/>'
                        html += f'<img src="data:image/jpeg;base64,{heatmap_base64}" alt="Heatmap"/>'  # Add heatmap
                        html += f'<p>Similarity: {similarity_score:.2f}</p>'

                        # Add a checkbox for removing the face
                        face_id = f'ref_{ref_idx}_{ref_face_idx}_sur_{sur_idx}_{sur_face_idx}'
                        html += f'<input type="checkbox" id="remove_{face_id}" name="selected_faces" value="{comparison_id}">'
                        html += f'<label for="remove_{face_id}">Remove</label>'

                        html += '</div>'
            html += '</div>'  # Close face-comparison-grid
            html += '</div>'  # Close matrix-cell
    html += '</div>'  # Close comparison-matrix
    return html


@app.route("/", methods=['GET', 'POST'])
def upload_images():
    logging.info("Route / hit")  # ADDED LOGGING
    scale_factor = DEFAULT_SCALE_FACTOR
    min_neighbors = DEFAULT_MIN_NEIGHBORS
    min_size = DEFAULT_MIN_SIZE
    similarity_threshold = DEFAULT_SIMILARITY_THRESHOLD

    reference_files = []
    surveillance_files = []

    if request.method == 'POST':
        reference_files = request.files.getlist('referenceImage')
        surveillance_files = request.files.getlist('surveillanceImage')

        # Get parameter values from the form
        try:
            scale_factor = float(request.form.get('scaleFactor', DEFAULT_SCALE_FACTOR))
            min_neighbors = int(request.form.get('minNeighbors', DEFAULT_MIN_NEIGHBORS))
            min_size = int(request.form.get('minSize', DEFAULT_MIN_SIZE))
            similarity_threshold = float(request.form.get('similarityThreshold', DEFAULT_SIMILARITY_THRESHOLD))
        except ValueError:
            logging.error("Invalid parameter value received from form.")
            # Handle the error, maybe set to default values or show an error message

        # Save files to the server and store paths in the session
        reference_image_paths = []
        for file in reference_files:
            # Generate a unique filename
            filename = str(uuid.uuid4()) + "_" + file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            reference_image_paths.append(filepath)

        surveillance_image_paths = []
        for file in surveillance_files:
            # Generate a unique filename
            filename = str(uuid.uuid4()) + "_" + file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            surveillance_image_paths.append(filepath)

        session['reference_image_paths'] = reference_image_paths
        session['surveillance_image_paths'] = surveillance_image_paths

        session['scale_factor'] = scale_factor
        session['min_neighbors'] = min_neighbors
        session['min_size'] = min_size
        session['similarity_threshold'] = similarity_threshold

        # Initialize or retrieve removed_face_ids from session
        if 'removed_face_ids' not in session:
            session['removed_face_ids'] = []
        session.modified = True

    # Retrieve image paths from session if available
    reference_image_paths = session.get('reference_image_paths', [])
    surveillance_image_paths = session.get('surveillance_image_paths', [])

    scale_factor = session.get('scale_factor', DEFAULT_SCALE_FACTOR)
    min_neighbors = session.get('min_neighbors', DEFAULT_MIN_NEIGHBORS)
    min_size = session.get('min_size', DEFAULT_MIN_SIZE)
    similarity_threshold = session.get('similarity_threshold', DEFAULT_SIMILARITY_THRESHOLD)

    removed_face_ids = session.get('removed_face_ids', [])

    comparison_matrix_html = ""
    if reference_image_paths and surveillance_image_paths:
        comparison_matrix_html = create_comparison_matrix(reference_image_paths, surveillance_image_paths,
                                                          scale_factor,
                                                          min_neighbors, min_size, similarity_threshold,
                                                          removed_face_ids)

    # Render the template with the comparison matrix
    final_html = HTML_TEMPLATE_START.format(scale_factor=scale_factor, min_neighbors=min_neighbors,
                                            min_size=min_size,
                                            similarity_threshold=similarity_threshold) + comparison_matrix_html + HTML_TEMPLATE_END
    return final_html


@app.route('/remove_faces', methods=['POST'])
def remove_faces():
    logging.info("Route /remove_faces hit")  # ADDED LOGGING
    """Handles the removal of selected faces from the comparison matrix."""
    removed_face_ids = json.loads(request.form.get('selected_face_ids'))
    scale_factor = float(request.form.get('scaleFactor'))
    min_neighbors = int(request.form.get('minNeighbors'))
    min_size = int(request.form.get('minSize'))
    similarity_threshold = float(request.form.get('similarityThreshold'))

    # Get image paths from session
    reference_image_paths = session.get('reference_image_paths', [])
    surveillance_image_paths = session.get('surveillance_image_paths', [])

    # Update the session with the removed face IDs
    if 'removed_face_ids' not in session:
        session['removed_face_ids'] = []
    session['removed_face_ids'].extend(removed_face_ids)
    session.modified = True

    # Regenerate the comparison matrix HTML
    comparison_matrix_html = create_comparison_matrix(reference_image_paths, surveillance_image_paths,
                                                      scale_factor,
                                                      min_neighbors, min_size, similarity_threshold,
                                                      session['removed_face_ids'])

    # Return only the updated comparison matrix HTML
    return comparison_matrix_html


if __name__ == "__main__":
    logging.info("App starting in debug mode")  # ADDED LOGGING
    app.run(debug=True)
