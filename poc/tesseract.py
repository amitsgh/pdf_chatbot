import pytesseract
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from PIL import Image
import io

# Path to your Spanish PDF
pdf_path = "poc/Spanish.pdf"


# Function to extract text and bounding boxes from each page using Tesseract
def extract_text_with_tesseract(pdf_path):
    images = convert_from_path(pdf_path)  # Convert PDF to images for OCR
    ocr_results = []

    for page_num, image in enumerate(images):
        # Use pytesseract to perform OCR on the image (Spanish language)
        text = pytesseract.image_to_string(image, lang='spa')
        
        # Get the bounding boxes (coordinates of detected text regions)
        boxes = pytesseract.image_to_boxes(image, lang='spa')
        
        ocr_results.append({
            'page_num': page_num + 1,
            'text': text,
            'boxes': boxes
        })

    return ocr_results, images


# Function to create a new PDF with highlighted OCR text
def create_highlighted_pdf(ocr_results, images, output_pdf_path):
    # Open the original PDF with PyMuPDF for editing
    doc = fitz.open(pdf_path)
    
    for i, result in enumerate(ocr_results):
        page = doc.load_page(i)
        image = images[i]

        # Extract bounding boxes for each character and highlight them
        for box in result['boxes'].splitlines():
            char, x1, y1, x2, y2 = box.split()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Add highlight (yellow) on the text region
            page.add_highlight_annot(fitz.Rect(x1, y1, x2, y2))
        
        # Optionally: You can also draw rectangles manually if you prefer
        
    # Save the modified PDF with highlights
    doc.save(output_pdf_path)
    print(f"Highlighted PDF saved at: {output_pdf_path}")


# Run OCR and generate the highlighted PDF
ocr_results, images = extract_text_with_tesseract(pdf_path)

# Output path for the new PDF with highlights
output_pdf_path = '/mnt/data/Highlighted_Spanish.pdf'

# Create the highlighted PDF
create_highlighted_pdf(ocr_results, images, output_pdf_path)

# Return the path to the newly created PDF
output_pdf_path
