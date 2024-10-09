import cv2
import pytesseract
import re
import numpy as np
from matplotlib import pyplot as plt

def extract_text_and_boxes(image):
    """Extracts text and bounding boxes from the image using pytesseract with config."""
    custom_config = r'--oem 3 --psm 6'  # OEM 3 for best accuracy, PSM 6 assumes single block of text
    data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
    return data

def combine_potential_email_parts(data):
    """Combines adjacent text components that might form an email address."""
    num_boxes = len(data['text'])
    combined_words = []
    combined_boxes = []
    i = 0
    
    while i < num_boxes:
        current_word = data['text'][i].strip()
        if not current_word:
            i += 1
            continue
            
        # Initialize the combined word and its bounding box
        combined_word = current_word
        left = data['left'][i]
        top = min(data['top'][i], data['top'][i] if i < num_boxes else float('inf'))
        right = data['left'][i] + data['width'][i]
        bottom = max(data['top'][i] + data['height'][i], 
                    data['top'][i] + data['height'][i] if i < num_boxes else 0)
        
        # Look ahead for potential email parts
        j = i + 1
        while j < num_boxes:
            next_word = data['text'][j].strip()
            if not next_word:
                j += 1
                continue
                
            # Check if the next component is close enough horizontally
            horizontal_gap = data['left'][j] - right
            if horizontal_gap > 50:  # Maximum allowed gap between components
                break
                
            # Check if the components are roughly on the same line
            vertical_overlap = (min(data['top'][i] + data['height'][i], data['top'][j] + data['height'][j]) -
                              max(data['top'][i], data['top'][j]))
            if vertical_overlap < 0:
                break
                
            # Combine the words
            combined_word += next_word
            right = data['left'][j] + data['width'][j]
            top = min(top, data['top'][j])
            bottom = max(bottom, data['top'][j] + data['height'][j])
            
            j += 1
        
        # Store the combined word and its bounding box
        combined_words.append(combined_word)
        combined_boxes.append({
            'left': left,
            'top': top,
            'width': right - left,
            'height': bottom - top
        })
        
        i = j if j > i + 1 else i + 1
    
    return combined_words, combined_boxes

def clean_email(word):
    """Cleans up potential email addresses by removing artifacts and normalizing characters."""
    # Remove common OCR artifacts and normalize
    word = re.sub(r'[€¢]', '@', word)  # Replace common @ symbol mistakes
    word = re.sub(r'\s+', '', word)    # Remove any whitespace
    word = re.sub(r'[^\w@.-]', '', word)  # Remove any other invalid characters
    return word

def is_email(word):
    """Enhanced email detection that handles common OCR errors."""
    # Clean the word first
    word = clean_email(word)
    
    # Basic email pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Alternative pattern for cases where @ might be missing or misrecognized
    alternative_pattern = r'^[a-zA-Z0-9._%+-]+[@€¢]+[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    return bool(re.match(email_pattern, word) or re.match(alternative_pattern, word))

def preprocess_image(image):
    """Enhances the image for better OCR detection."""
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    binary_image = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size
        2    # Constant subtracted from mean
    )

    # Apply dilation and erosion to remove noise
    kernel = np.ones((1, 1), np.uint8)
    processed_image = cv2.dilate(binary_image, kernel, iterations=1)
    processed_image = cv2.erode(processed_image, kernel, iterations=1)
    
    return processed_image

def blur_email_regions(image, data):
    """Modified function to handle split email addresses."""
    # First, combine potential email parts
    combined_words, combined_boxes = combine_potential_email_parts(data)
    
    print("Recognized text components:")
    for word, box in zip(combined_words, combined_boxes):
        print(f"Combined word: '{word}', Box: {box}")
        
    emails_blurred = set()
    
    for word, box in zip(combined_words, combined_boxes):
        # Clean up the word
        cleaned_word = clean_email(word)
        
        if is_email(cleaned_word):
            if cleaned_word in emails_blurred:
                continue
                
            emails_blurred.add(cleaned_word)
            print(f"Blurring email: {cleaned_word}")
            
            # Get the bounding box coordinates
            x, y, w, h = (box['left'], box['top'], box['width'], box['height'])
            
            # Add padding to ensure full email is blurred
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2*padding)
            h = min(image.shape[0] - y, h + 2*padding)
            
            # Extract and blur the region
            roi = image[y:y+h, x:x+w]
            blurred_roi = cv2.GaussianBlur(roi, (25, 25), 50)
            image[y:y+h, x:x+w] = blurred_roi
    
    return image, list(emails_blurred)

def visualize_text_boxes(image, data):
    """Draws bounding boxes around recognized text for visualization."""
    num_boxes = len(data['text'])
    image_copy = image.copy()

    for i in range(num_boxes):
        word = data['text'][i]
        if word.strip():
            x, y, w, h = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_copy, word, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert BGR to RGB for display in matplotlib
    image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 15))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title('Detected Text Regions')
    plt.show()

def process_image(image_path):
    """Main function to process the image and blur emails."""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Create a copy for visualization
    original_image = image.copy()

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Extract text and bounding boxes
    data = extract_text_and_boxes(preprocessed_image)

    # Visualize original text detection
    print("Original text detection:")
    visualize_text_boxes(original_image, data)

    # Blur email regions
    blurred_image, detected_emails = blur_email_regions(image, data)

    # Display results
    print("\nDetected and blurred emails:")
    for email in detected_emails:
        print(f"- {email}")

    # Display the final blurred image
    plt.figure(figsize=(15, 15))
    plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Final Image with Blurred Emails')
    plt.show()

    return blurred_image, detected_emails

# If running as a script
if __name__ == "__main__":
    # Replace with your image path
    image_path = "Screenshot_2.png"
    try:
        blurred_image, detected_emails = process_image(image_path)
        print(f"\nSuccessfully processed image. {len(detected_emails)} emails were detected and blurred.")
    except Exception as e:
        print(f"Error processing image: {str(e)}")