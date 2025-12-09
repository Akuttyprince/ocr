import io
import os
import re
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image, ImageStat, ImageEnhance, ImageFilter
import cv2
import numpy as np
from difflib import SequenceMatcher
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse

app = FastAPI(title="OCR Web (Tesseract OCR)")



def safe_group(match, group_num=1):
    """Safely access a regex capture group."""
    try:
        return match.group(group_num) if match and match.groups() and len(match.groups()) >= group_num else None
    except (IndexError, AttributeError):
        return None


def _enhance_image_for_ocr(img):
    """Enhanced image preprocessing for better OCR on low quality images."""
    import cv2
    import numpy as np

    try:
        # Convert PIL to OpenCV format
        img_array = np.array(img)

        # Ensure we have a 3-channel image
        if len(img_array.shape) == 2:
            # Already grayscale, convert to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # RGBA, convert to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # 1. Resize image for better OCR (increase size if too small)
        height, width = gray.shape
        min_dimension = 1000
        if max(height, width) < min_dimension:
            scale_factor = min_dimension / max(height, width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # 2. Apply bilateral filter for edge-preserving smoothing
        try:
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
        except:
            pass

        # 3. Multiple pass enhancement
        enhanced = gray.copy()

        # First pass: Denoising
        try:
            denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)
            enhanced = denoised
        except:
            try:
                # Fallback to Gaussian blur
                enhanced = cv2.GaussianBlur(enhanced, (3,3), 0)
            except:
                pass

        # Second pass: Contrast enhancement
        try:
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(enhanced)
        except:
            # Fallback to histogram equalization
            try:
                enhanced = cv2.equalizeHist(enhanced)
            except:
                pass

        # Third pass: Sharpening with unsharp mask
        try:
            # Create unsharp mask
            blurred = cv2.GaussianBlur(enhanced, (0,0), 2.0)
            sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
            enhanced = np.clip(sharpened, 0, 255)
        except:
            # Fallback to basic sharpening
            try:
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(enhanced, -1, kernel)
                enhanced = np.clip(sharpened, 0, 255)
            except:
                pass

        # 4. Adaptive thresholding with better parameters
        try:
            # Try adaptive threshold first
            binary = cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                15,
                8
            )

            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Remove small noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        except:
            # Fallback to Otsu thresholding
            try:
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            except:
                # If all fails, return enhanced image
                binary = enhanced

        # Convert back to PIL Image (RGB)
        if len(binary.shape) == 2:
            # Grayscale to RGB
            processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        else:
            processed = binary

        # Additional cleanup: invert if needed (OCR usually works better on dark text on light background)
        mean_val = np.mean(processed)
        if mean_val < 127:
            processed = 255 - processed

        return Image.fromarray(processed.astype(np.uint8))

    except Exception as e:
        # If everything fails, return the original image
        return img


def _auto_enhance_brightness(img):
    """Automatically enhance brightness and contrast if image is too dark or too bright."""
    from PIL import ImageEnhance, ImageStat
    # Convert to grayscale to measure brightness
    gray = img.convert('L')
    stat = ImageStat.Stat(gray)
    mean_brightness = stat.mean[0]  # 0-255

    # Calculate standard deviation for contrast measurement
    std_brightness = stat.stddev[0]

    # Enhanced image
    enhanced_img = img.copy()

    # 1. Enhance brightness if too dark
    if mean_brightness < 100:
        # Very dark image - aggressive enhancement
        brightness_factor = min(3.0, 180 / (mean_brightness + 1))
        enhancer = ImageEnhance.Brightness(enhanced_img)
        enhanced_img = enhancer.enhance(brightness_factor)
    elif mean_brightness > 200:
        # Too bright - reduce brightness
        brightness_factor = 0.8
        enhancer = ImageEnhance.Brightness(enhanced_img)
        enhanced_img = enhancer.enhance(brightness_factor)

    # 2. Enhance contrast based on standard deviation
    if std_brightness < 50:  # Low contrast image
        # Increase contrast significantly
        contrast_factor = min(2.5, 100 / (std_brightness + 1))
        enhancer = ImageEnhance.Contrast(enhanced_img)
        enhanced_img = enhancer.enhance(contrast_factor)
    elif std_brightness > 120:  # High contrast - reduce slightly
        contrast_factor = 0.9
        enhancer = ImageEnhance.Contrast(enhanced_img)
        enhanced_img = enhancer.enhance(contrast_factor)

    # 3. Apply sharpening to improve text clarity
    enhancer = ImageEnhance.Sharpness(enhanced_img)
    enhanced_img = enhancer.enhance(1.2)

    return enhanced_img




def _setup_tesseract():
    """Ensure tesseract binary and tessdata are configured before calling pytesseract."""
    try:
        import pytesseract
    except Exception as e:
        raise RuntimeError("pytesseract not installed") from e
    
    # Ensure tesseract binary is available: try to detect common install locations on Windows
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        possible = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.join(os.path.expanduser("~"), "scoop", "shims", "tesseract.exe"),
            os.path.join(os.path.expanduser("~"), "scoop", "apps", "tesseract", "current", "tesseract.exe"),
            os.path.join(os.path.expanduser("~"), "scoop", "apps", "tesseract", "current", "bin", "tesseract.exe"),
        ]
        for p in possible:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break

    # Make sure TESSDATA_PREFIX is set (scoop/persist location is a common place)
    tessdata_candidates = [
        os.path.join(os.path.expanduser("~"), "scoop", "persist", "tesseract", "tessdata"),
        os.path.join(os.path.expanduser("~"), "scoop", "apps", "tesseract", "current", "tessdata"),
    ]
    for td in tessdata_candidates:
        if os.path.isdir(td):
            os.environ.setdefault("TESSDATA_PREFIX", td)
            break


def _pytesseract_text_and_conf(image):
    """Run pytesseract and return (text, avg_confidence)."""
    import pytesseract
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = data.get('text', [])
    confs = data.get('conf', [])
    texts = [w for w in words if w and w.strip()]
    valid_confs = [int(c) for c in confs if c not in (None, '', '-1')]
    avg_conf = float(sum(valid_confs)) / len(valid_confs) if valid_confs else 0.0
    text = " ".join(texts).strip()
    if not text:
        text = pytesseract.image_to_string(image)
    return text, avg_conf


def scan_with_tesseract_bytes(content: bytes):
    try:
        from PIL import Image
        import pytesseract
        from io import BytesIO
    except Exception as e:
        raise RuntimeError("pytesseract / pillow not installed") from e

    _setup_tesseract()
    img = Image.open(BytesIO(content))

    # Try advanced image enhancement first, fall back to basic if it fails
    try:
        img = _enhance_image_for_ocr(img)
    except:
        img = _auto_enhance_brightness(img)

    # Multiple OCR attempts with different configurations
    texts = []

    # Configuration 1: Best for general text
    try:
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ,.-/()&:;@$%&\'?!"'
        text1 = pytesseract.image_to_string(img, config=custom_config)
        if text1.strip():
            texts.append(text1.strip())
    except:
        pass

    # Configuration 2: Best for numbers and structured data
    try:
        custom_config = r'--oem 3 --psm 4'
        text2 = pytesseract.image_to_string(img, config=custom_config)
        if text2.strip():
            texts.append(text2.strip())
    except:
        pass

    # Configuration 3: Default configuration
    try:
        text3 = pytesseract.image_to_string(img)
        if text3.strip():
            texts.append(text3.strip())
    except:
        pass

    # Configuration 4: Sparse text (for passbooks)
    try:
        custom_config = r'--oem 3 --psm 11'
        text4 = pytesseract.image_to_string(img, config=custom_config)
        if text4.strip():
            texts.append(text4.strip())
    except:
        pass

    # Return the longest text (usually has the most content)
    if texts:
        return max(texts, key=len)
    else:
        return ""


def ensemble_tesseract_bytes(content: bytes, runs: int = 3):
    """Run OCR multiple times (default 3) on the same image and return the most common result."""
    try:
        from PIL import Image
        import pytesseract
        from io import BytesIO
        from collections import Counter
    except Exception as e:
        raise RuntimeError("pytesseract / pillow not installed") from e

    _setup_tesseract()
    img = Image.open(BytesIO(content))

    # Try advanced image enhancement first, fall back to basic if it fails
    try:
        img = _enhance_image_for_ocr(img)
    except:
        img = _auto_enhance_brightness(img)

    results = []
    for _ in range(runs):
        try:
            text, conf = _pytesseract_text_and_conf(img)
        except Exception:
            text, conf = "", 0.0
        results.append((text, conf))

    # Majority vote: pick most common text
    texts = [t for t, _ in results if t]
    if not texts:
        return ""

    text_counts = Counter(texts)
    most_common_text, count = text_counts.most_common(1)[0]

    # If all results are different (count == 1 for all), pick highest confidence
    if count == 1 and len(text_counts) == len(texts):
        best = max(results, key=lambda x: x[1])
        return best[0]

    return most_common_text


def detect_document_type(text: str) -> str:
    """Detect if the document is Aadhaar, PAN, or Bank Passbook."""
    text_lower = text.lower()

    # Check for Bank Passbook/Statement FIRST (before Aadhaar)
    # This is important because bank identity cards can contain text that triggers Aadhaar detection
    bank_indicators = [
        'bank', 'passbook', 'account statement', 'branch',
        'ifsc', 'micr', 'deposit', 'withdrawal', 'balance',
        'banking', 'बैंक', 'खाता', 'account number', 'kiosk banking',
        'cif number', 'identity card'  # Bank identity cards
    ]
    # Count bank indicators
    bank_score = sum(1 for indicator in bank_indicators if indicator in text_lower)
    
    # Check for PAN card
    pan_indicators = [
        'permanent account number', 'income tax department',
        'pan', 'आयकर विभाग', 'income tax'
    ]
    pan_pattern = r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b'
    pan_score = sum(1 for indicator in pan_indicators if indicator in text_lower)
    if re.search(pan_pattern, text):
        pan_score += 2  # Strong indicator
    
    # Check for Aadhaar card
    aadhaar_indicators = [
        'aadhaar', 'uidai', 'unique identification',
        'government of india', 'आधार', 'uid'
    ]
    aadhaar_pattern = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
    aadhaar_score = sum(1 for indicator in aadhaar_indicators if indicator in text_lower)
    if re.search(aadhaar_pattern, text):
        aadhaar_score += 1  # Weak indicator (12-digit numbers are common)
    
    # Determine document type based on scores
    if bank_score >= 2:  # At least 2 bank indicators
        return "bank"
    elif pan_score >= 2:
        return "pan"
    elif aadhaar_score >= 2:
        return "aadhaar"
    elif bank_score > 0:
        return "bank"
    elif pan_score > 0:
        return "pan"
    elif aadhaar_score > 0:
        return "aadhaar"

    return "general"

def extract_aadhaar_details(text: str) -> Dict[str, Any]:
    details = {
        "document_type": "Aadhaar Card",
        "name": None,
        "aadhaar_number": None,
        "date_of_birth": None,
        "gender": None,
        "address": None,
        "pincode": None,
        "mobile": None,
        "enrollment_number": None
    }

    # Clean + split lines
    lines = [l.strip() for l in text.replace("\r", "").split("\n") if l.strip()]
    joined = " ".join(lines)

    # Aadhaar number
    aadhaar_match = re.search(r'(\d{4}\s?\d{4}\s?\d{4})', joined)
    if aadhaar_match:
        aadhaar_num = safe_group(aadhaar_match, 1)
        if aadhaar_num:
            details["aadhaar_number"] = aadhaar_num.replace(" ", "")

    # DOB / Year of Birth
    dob = None
    yob = None
    for line in lines:
        m = re.search(r'(DOB|D\.O\.B|Date of Birth)[:\s]*([0-3]?\d[-/][01]?\d[-/][12]\d{3})', line, re.IGNORECASE)
        if m:
            dob = m.group(2)
            break
        y = re.search(r'Year of Birth[:\s]*([12]\d{3})', line, re.IGNORECASE)
        if y:
            yob = y.group(1)

    if dob:
        details["date_of_birth"] = dob
    elif yob:
        details["date_of_birth"] = yob  # fallback as year only

    # Name extraction - multiple strategies
    name = None
    
    # Strategy 1: Look for line before DOB
    dob_idx = -1
    for idx, line in enumerate(lines):
        if re.search(r'(DOB|D\.O\.B|Date of Birth|Year of Birth)', line, re.IGNORECASE):
            dob_idx = idx
            break

    if dob_idx > 0:
        name_candidate = lines[dob_idx - 1]
        # Skip lines with S/O, D/O, W/O
        if re.search(r'(S/O|D/O|W/O)', name_candidate, re.IGNORECASE) and dob_idx > 1:
            name_candidate = lines[dob_idx - 2]
        # Clean and validate
        cleaned = re.sub(r'[^A-Za-z\s]', ' ', name_candidate)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if len(cleaned.split()) >= 2:  # At least 2 words
            name = cleaned
    
    # Strategy 2: Look for explicit "Name" label
    if not name:
        for i, line in enumerate(lines):
            if re.search(r'^\s*(Name|नाम)[:\s]', line, re.IGNORECASE):
                # Check same line
                m = re.search(r'(Name|नाम)[:\s]+([A-Za-z\s]{3,})', line, re.IGNORECASE)
                if m and safe_group(m, 2):
                    name = safe_group(m, 2).strip()
                # Check next line
                elif i + 1 < len(lines):
                    candidate = lines[i + 1]
                    cleaned = re.sub(r'[^A-Za-z\s]', ' ', candidate)
                    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                    if len(cleaned.split()) >= 2:
                        name = cleaned
                break
    
    # Strategy 3: Find first line with 2+ capitalized words (likely a name)
    if not name:
        for line in lines[:10]:  # Check first 10 lines
            # Skip lines with common keywords
            if re.search(r'(government|india|aadhaar|uidai|card|number)', line, re.IGNORECASE):
                continue
            # Look for capitalized words pattern
            words = line.split()
            cap_words = [w for w in words if w and w[0].isupper() and len(w) > 2 and w.isalpha()]
            if len(cap_words) >= 2:
                name = ' '.join(cap_words)
                break
    
    if name:
        # Final cleaning
        name = re.sub(r'\s+', ' ', name).strip().title()
        details["name"] = name

    # Gender detection
    for line in lines:
        if re.search(r'\b(Male|पुरुष)\b', line, re.IGNORECASE):
            details["gender"] = "Male"
            break
        if re.search(r'\b(Female|महिला)\b', line, re.IGNORECASE):
            details["gender"] = "Female"
            break

    # Address extraction - combine lines after name/DOB until pincode
    address_lines = []
    start_collecting = False
    
    for i, line in enumerate(lines):
        # Start after name or DOB
        if name and name.lower() in line.lower():
            start_collecting = True
            continue
        elif re.search(r'(DOB|Date of Birth|Year of Birth)', line, re.IGNORECASE):
            start_collecting = True
            continue
        
        if start_collecting:
            # Stop at certain keywords
            if re.search(r'(enrollment|vid|download|generated|issued)', line, re.IGNORECASE):
                break
            # Clean and add line
            cleaned = line.strip()
            if cleaned and len(cleaned) > 3:
                address_lines.append(cleaned)
            # Stop after collecting enough lines or finding pincode
            if len(address_lines) >= 5 or re.search(r'\b\d{6}\b', line):
                break
    
    if address_lines:
        full_address = ' '.join(address_lines)
        # Clean up
        full_address = re.sub(r'\s+', ' ', full_address).strip()
        if len(full_address) > 10:
            details["address"] = full_address

    # PIN code
    pin_match = re.search(r'\b(\d{6})\b', joined)
    if pin_match:
        details["pincode"] = safe_group(pin_match, 1)

    # TODO: mobile, enrollment, address (can add later)
    return {k: v for k, v in details.items() if v is not None}


def extract_pan_details(text: str) -> Dict[str, Any]:
    """Extract structured data from PAN card text with improved accuracy."""
    details = {}
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    joined = ' '.join(lines)
    
    # PAN number extraction
    pan_patterns = [
        r'\b([A-Z]{5}[0-9]{4}[A-Z]{1})\b',
        r'Permanent\s+Account\s+Number[:\s]*([A-Z]{5}[0-9]{4}[A-Z]{1})',
        r'PAN[:\s]*([A-Z]{5}[0-9]{4}[A-Z]{1})'
    ]
    for pattern in pan_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.groups():
            pan_num = safe_group(match, 1)
            if pan_num:
                details["pan_number"] = pan_num.upper()
            break
    
    # NAME EXTRACTION - 5 strategies
    name = None
    
    # Strategy 1: Look for all-caps name (2-4 words, all uppercase)
    for line in lines:
        # Skip common headers
        if re.search(r'(INCOME TAX|GOVERNMENT|INDIA|DEPARTMENT|PERMANENT|ACCOUNT)', line, re.IGNORECASE):
            continue
        # Look for all-caps names (2-4 words)
        words = line.split()
        if 2 <= len(words) <= 4:
            if all(w.isupper() and w.isalpha() and len(w) > 1 for w in words):
                name = ' '.join(words).title()
                break
    
    # Strategy 2: Look for "Name" label
    if not name:
        name_patterns = [
            r'(?:Name|नाम)[:\s]+([A-Z][A-Za-z\s]{5,50})',
            r'\n([A-Z][A-Z\s]{10,40})\n',  # All caps on own line
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                candidate = safe_group(match, 1)
                if candidate:
                    candidate = candidate.strip()
                    # Skip if it's a header
                    if not re.search(r'(INCOME|TAX|GOVERNMENT|INDIA|DEPARTMENT)', candidate, re.IGNORECASE):
                        cleaned = re.sub(r'[^A-Za-z\s]', ' ', candidate)
                        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                        words = cleaned.split()
                        if 2 <= len(words) <= 5:
                            name = cleaned.title()
                            break
    
    # Strategy 3: Look after PAN number
    if not name and "pan_number" in details:
        pan_num = details["pan_number"]
        # Find position of PAN number
        pan_pos = text.find(pan_num)
        if pan_pos > 0:
            # Look in text before PAN number
            before_pan = text[:pan_pos]
            before_lines = [l.strip() for l in before_pan.split('\n') if l.strip()]
            # Check last few lines before PAN
            for line in reversed(before_lines[-5:]):
                if re.search(r'(INCOME|TAX|GOVERNMENT|INDIA|DEPARTMENT|PERMANENT|ACCOUNT)', line, re.IGNORECASE):
                    continue
                words = line.split()
                if 2 <= len(words) <= 4:
                    # Check if mostly alphabetic
                    alpha_words = [w for w in words if re.match(r'^[A-Za-z]+$', w) and len(w) > 1]
                    if len(alpha_words) >= 2:
                        name = ' '.join(alpha_words).title()
                        break
    
    # Strategy 4: Look for pattern "Father's Name" and take line before it
    if not name:
        for i, line in enumerate(lines):
            if re.search(r"(Father|Father's|पिता)", line, re.IGNORECASE):
                if i > 0:
                    candidate = lines[i-1]
                    if not re.search(r'(INCOME|TAX|GOVERNMENT|INDIA)', candidate, re.IGNORECASE):
                        cleaned = re.sub(r'[^A-Za-z\s]', ' ', candidate)
                        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                        words = cleaned.split()
                        if 2 <= len(words) <= 5:
                            name = cleaned.title()
                            break
    
    # Strategy 5: Find largest all-caps text block (likely the name)
    if not name:
        caps_blocks = re.findall(r'\b([A-Z][A-Z\s]{10,40})\b', text)
        for block in caps_blocks:
            if re.search(r'(INCOME|TAX|GOVERNMENT|INDIA|DEPARTMENT|PERMANENT|ACCOUNT)', block):
                continue
            words = block.split()
            if 2 <= len(words) <= 4:
                name = block.title()
                break
    
    if name:
        details["name"] = name
    
    # FATHER'S NAME EXTRACTION
    father_patterns = [
        r"(?:Father's Name|Father Name|पिता का नाम)[:\s]*([A-Z][A-Za-z\s.]{3,50})",
        r"(?:Father|पिता)[:\s]*([A-Z][A-Za-z\s.]{3,50})"
    ]
    for pattern in father_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            father = safe_group(match, 1)
            if father:
                father = father.strip()
                # Clean
                father = re.sub(r'[^A-Za-z\s.]', ' ', father)
                father = re.sub(r'\s+', ' ', father).strip()
                if len(father.split()) >= 2:
                    details["father_name"] = father.title()
                break
    
    # DOB EXTRACTION - 4 strategies
    dob = None
    
    # Strategy 1: Look for labeled DOB
    dob_patterns = [
        r'(?:Date of Birth|DOB|D\.O\.B|जन्म तिथि)[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})',
        r'(?:Date of Birth|DOB|D\.O\.B|जन्म तिथि)[:\s]*(\d{2}[/-]\d{2}[/-]\d{2})',
    ]
    for pattern in dob_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            dob = safe_group(match, 1)
            if dob:
                details["date_of_birth"] = dob
                break
    
    # Strategy 2: Look for date pattern near "Birth" keyword
    if not dob:
        birth_context = re.search(r'Birth.{0,30}(\d{2}[/-]\d{2}[/-]\d{4})', text, re.IGNORECASE)
        if birth_context:
            dob = safe_group(birth_context, 1)
            if dob:
                details["date_of_birth"] = dob
    
    # Strategy 3: Look for any date pattern (DD/MM/YYYY or DD-MM-YYYY)
    if not dob:
        date_match = re.search(r'\b(\d{2}[/-]\d{2}[/-]\d{4})\b', text)
        if date_match:
            dob = safe_group(date_match, 1)
            if dob:
                # Validate it's a reasonable date (not account number etc)
                parts = re.split(r'[/-]', dob)
                if len(parts) == 3:
                    day, month, year = parts
                    if 1 <= int(day) <= 31 and 1 <= int(month) <= 12:
                        details["date_of_birth"] = dob
    
    # Strategy 4: Look for year only (4 digits between 1900-2020)
    if not dob:
        year_match = re.search(r'\b(19\d{2}|20[0-2]\d)\b', text)
        if year_match:
            year = safe_group(year_match, 1)
            if year:
                details["date_of_birth"] = year
    
    return {k: v for k, v in details.items() if v is not None}


def extract_bank_details(text: str) -> Dict[str, Any]:
    details = {
        "document_type": "Bank Passbook/Statement",
        "account_holder_name": None,
        "account_number": None,
        "bank_name": None,
        "branch_name": None,
        "ifsc_code": None,
        "micr_code": None,
        "balance": None,
        "address": None,
        "contact": None
    }

    lines = [l.strip() for l in text.replace("\r", "").split("\n") if l.strip()]
    joined = " ".join(lines)

    # Bank name – first line with 'BANK'
    for line in lines:
        if "bank" in line.lower():
            details["bank_name"] = line.strip()
            break

    # Account number – try labeled, else first long digit group
    m = re.search(r'(Account\s*No\.?|A/c\s*No\.?|A/C\s*NO\.?)[:\s]*([0-9\- ]{8,20})', joined, re.IGNORECASE)
    if m:
        acc = safe_group(m, 2)
        if acc:
            details["account_number"] = re.sub(r'\D', '', acc)
    else:
        m2 = re.search(r'\b(\d{9,18})\b', joined)
        if m2:
            acc_num = safe_group(m2, 1)
            if acc_num:
                details["account_number"] = acc_num

    # IFSC - more flexible pattern to handle OCR errors
    m = re.search(r'([A-Z0-9]{4}[0O][A-Z0-9]{6})', joined, re.IGNORECASE)
    if m:
        ifsc = safe_group(m, 1)
        if ifsc:
            details["ifsc_code"] = ifsc.upper()

    # MICR
    m = re.search(r'MICR[:\s]*(\d{9})', joined, re.IGNORECASE)
    if m:
        micr = safe_group(m, 1)
        if micr:
            details["micr_code"] = micr

    # Account holder name - improved extraction
    holder_name = None
    
    # Strategy 1: Look for labeled fields
    for i, line in enumerate(lines):
        # Check for name labels
        if re.search(r'(Account\s*Holder|Customer\s*Name|Name|First\s*Name)', line, re.IGNORECASE):
            # Same line value
            m = re.search(r':\s*([A-Za-z][A-Za-z\s.]{2,50})', line, re.IGNORECASE)
            if m:
                candidate = safe_group(m, 1)
                if candidate:
                    holder_name = candidate.strip().title()
                    break
            # Next line value
            elif i + 1 < len(lines):
                candidate = lines[i+1]
                cleaned = re.sub(r'[^A-Za-z\s]', ' ', candidate)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                if len(cleaned.split()) >= 2:
                    holder_name = cleaned.title()
                    break
    
    # Strategy 2: Combine First + Middle + Last name fields
    if not holder_name:
        first_name = None
        middle_name = None
        last_name = None
        
        for i, line in enumerate(lines):
            if re.search(r'First\s*Name', line, re.IGNORECASE):
                # Check same line or next line
                m = re.search(r':\s*([A-Za-z]+)', line, re.IGNORECASE)
                if m:
                    first_name = safe_group(m, 1)
                elif i + 1 < len(lines):
                    candidate = re.sub(r'[^A-Za-z]', '', lines[i+1])
                    if candidate:
                        first_name = candidate
            
            if re.search(r'Middle\s*Name', line, re.IGNORECASE):
                m = re.search(r':\s*([A-Za-z]+)', line, re.IGNORECASE)
                if m:
                    middle_name = safe_group(m, 1)
                elif i + 1 < len(lines):
                    candidate = re.sub(r'[^A-Za-z]', '', lines[i+1])
                    if candidate and len(candidate) > 1:
                        middle_name = candidate
            
            if re.search(r'Last\s*Name', line, re.IGNORECASE):
                m = re.search(r':\s*([A-Za-z]+)', line, re.IGNORECASE)
                if m:
                    last_name = safe_group(m, 1)
                elif i + 1 < len(lines):
                    candidate = re.sub(r'[^A-Za-z]', '', lines[i+1])
                    if candidate:
                        last_name = candidate
        
        # Combine names
        if first_name and last_name:
            name_parts = [first_name]
            if middle_name:
                name_parts.append(middle_name)
            name_parts.append(last_name)
            holder_name = ' '.join(name_parts).title()
    
    if holder_name:
        details["account_holder_name"] = holder_name

    # Branch (simple)
    for line in lines:
        m = re.search(r'Branch[:\s]*(.+)', line, re.IGNORECASE)
        if m:
            branch = safe_group(m, 1)
            if branch:
                details["branch_name"] = branch.strip()
            break

    return {k: v for k, v in details.items() if v is not None}


def extract_structured_data(text: str) -> Dict[str, Any]:
    """Extract structured data based on document type."""
    # First detect the document type
    doc_type = detect_document_type(text)

    # Extract specific details based on document type
    if doc_type == "aadhaar":
        return extract_aadhaar_details(text)
    elif doc_type == "pan":
        return extract_pan_details(text)
    elif doc_type == "bank":
        return extract_bank_details(text)
    else:
        # General extraction for other documents
        return extract_general_data(text)


def extract_general_data(text: str) -> Dict[str, Any]:
    """Extract general structured data for non-Indian documents."""
    extracted_data = {
        "document_type": "General",
        "raw_text": text,
        "emails": [],
        "phone_numbers": [],
        "dates": [],
        "urls": [],
        "addresses": [],
        "numbers": [],
        "names": [],
        "organizations": [],
        "monetary_values": [],
        "identification_numbers": []
    }

    # Email patterns
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    extracted_data["emails"] = re.findall(email_pattern, text)

    # Phone number patterns (multiple formats)
    phone_patterns = [
        r'\b\d{3}-\d{3}-\d{4}\b',  # 123-456-7890
        r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',  # (123) 456-7890
        r'\b\+?1?\s*\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # +1 123 456 7890
        r'\b\d{10}\b'  # 1234567890
    ]
    for pattern in phone_patterns:
        extracted_data["phone_numbers"].extend(re.findall(pattern, text))
    extracted_data["phone_numbers"] = list(set(extracted_data["phone_numbers"]))

    # Date patterns
    date_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or DD/MM/YYYY
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY-MM-DD
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b',  # Month Day, Year
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*,?\s*\d{2,4}\b'  # Day Month Year
    ]
    for pattern in date_patterns:
        extracted_data["dates"].extend(re.findall(pattern, text))

    # URL patterns
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    extracted_data["urls"] = re.findall(url_pattern, text)

    # Address patterns (basic)
    address_patterns = [
        r'\b\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Way|Place|Pl)\b',
        r'\b\d+\s+[\w\s]+(?:#[\w-]+)?\b',  # Number + Street name with optional apartment number
        r'\b[A-Z]{2}\s*\d{5}\b'  # ZIP codes
    ]
    for pattern in address_patterns:
        extracted_data["addresses"].extend(re.findall(pattern, text))

    # Monetary values
    money_patterns = [
        r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?',  # $1,234.56
        r'\$\s*\d+(?:\.\d{2})?',  # $123.45
        r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|dollars?|cents?)\b',
        r'USD\s*\$\s*\d+(?:,\d{3})*(?:\.\d{2})?'
    ]
    for pattern in money_patterns:
        extracted_data["monetary_values"].extend(re.findall(pattern, text))

    # Identification numbers (SSN, ID, etc.)
    id_patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN format
        r'\bID\s*#?\s*\w+\b',
        r'\b(?:SSN|Social\s*Security)\s*:?\s*\d{3}-\d{2}-\d{4}\b'
    ]
    for pattern in id_patterns:
        extracted_data["identification_numbers"].extend(re.findall(pattern, text))

    # General numbers
    number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
    extracted_data["numbers"] = re.findall(number_pattern, text)

    # Potential names (Capitalized words - simplified)
    name_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
    extracted_data["names"] = re.findall(name_pattern, text)

    # Organizations (Capitalized words with Inc., LLC, etc.)
    org_pattern = r'\b[\w\s]+(?:Inc|LLC|Corp|Corporation|Company|Co|Ltd|Limited|PLC|GmbH|AG)\b'
    extracted_data["organizations"] = re.findall(org_pattern, text, re.IGNORECASE)

    return extracted_data


def choose_mode_and_scan(content: bytes, mode: str = "auto"):
    # Only Tesseract mode is available
    return scan_with_tesseract_bytes(content)


@app.get("/", response_class=HTMLResponse)
def index():
    html = Path(__file__).parent.parent.joinpath("frontend", "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.post("/upload")
async def upload(file: UploadFile = File(...), mode: str = Form(default="auto"), ensemble: bool = Form(default=False), runs: int = Form(default=3), structured: str = Form(default="true")):
    try:
        content = await file.read()
    except Exception as e:
        print(f"Error reading uploaded file: {str(e)}")
        return JSONResponse({"error": f"Failed to read file: {str(e)}"}, status_code=500)

    try:
        # Convert structured string to boolean
        structured_bool = structured.lower() in ["true", "on", "yes", "1"]

        if ensemble:
            # Use ensemble mode (3-pass by default)
            text = ensemble_tesseract_bytes(content, runs=runs)
        else:
            text = choose_mode_and_scan(content, mode)

        # Extract structured data from the OCR text
        if structured_bool:
            try:
                structured_data = extract_structured_data(text)
                # Add raw_text to structured data if it doesn't already have it
                if "raw_text" not in structured_data:
                    structured_data["raw_text"] = text
                return {
                    "filename": file.filename,
                    "structured_data": structured_data
                }
            except Exception as e:
                # If structured extraction fails, fall back to raw text
                print(f"Error in structured data extraction: {str(e)}")
                return {
                    "filename": file.filename,
                    "structured_data": {
                        "document_type": "General",
                        "raw_text": text,
                        "error": f"Structured extraction failed: {str(e)}"
                    }
                }
        else:
            return {"filename": file.filename, "raw_text": text}
    except Exception as e:
        tb = traceback.format_exc()
        # Return the error and traceback for easier debugging in dev
        return JSONResponse({"error": str(e), "traceback": tb}, status_code=500)


# ============================================================================
# VERIFICATION FUNCTIONS
# ============================================================================

def normalize_name(name: str) -> str:
    """Normalize name for comparison."""
    if not name:
        return ""
    name = name.lower()
    # Remove titles
    titles = ['mr', 'mrs', 'ms', 'dr', 'prof', 'shri', 'smt', 'kumari']
    words = name.split()
    words = [w for w in words if w.strip('.') not in titles]
    name = ' '.join(words)
    # Remove special characters except spaces
    name = re.sub(r'[^a-z\s]', '', name)
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def calculate_name_similarity(name1: str, name2: str) -> float:
    """Calculate similarity between two names (0.0 to 1.0)."""
    if not name1 or not name2:
        return 0.0
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    if not n1 or not n2:
        return 0.0
    # Full name comparison
    full_match = SequenceMatcher(None, n1, n2).ratio()
    # First + Last name comparison
    words1 = n1.split()
    words2 = n2.split()
    if len(words1) >= 2 and len(words2) >= 2:
        first_match = SequenceMatcher(None, words1[0], words2[0]).ratio()
        last_match = SequenceMatcher(None, words1[-1], words2[-1]).ratio()
        partial_match = (first_match + last_match) / 2
        return max(full_match, partial_match)
    return full_match


def normalize_dob(dob: str) -> Optional[str]:
    """Convert DOB to YYYY-MM-DD format."""
    if not dob:
        return None
    dob = str(dob).strip()
    formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
        "%d/%m/%y", "%d-%m-%y", "%d.%m.%y",
        "%m/%d/%Y", "%m-%d-%Y"
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(dob, fmt)
            return dt.strftime("%Y-%m-%d")
        except:
            continue
    if dob.isdigit() and len(dob) == 4:
        return f"{dob}-01-01"
    return None


def compare_dob(dob1: str, dob2: str) -> Tuple[bool, str]:
    """Compare two dates of birth."""
    if not dob1 or not dob2:
        return (None, "DOB not available in one or both documents")
    norm_dob1 = normalize_dob(dob1)
    norm_dob2 = normalize_dob(dob2)
    if not norm_dob1 or not norm_dob2:
        return (None, "Could not parse DOB format")
    if norm_dob1 == norm_dob2:
        return (True, f"DOB matches: {norm_dob1}")
    # Check if only year matches
    year1 = norm_dob1.split('-')[0]
    year2 = norm_dob2.split('-')[0]
    if year1 == year2 and (dob1.isdigit() or dob2.isdigit()):
        return (True, f"Birth year matches: {year1}")
    return (False, f"DOB mismatch: {norm_dob1} vs {norm_dob2}")


def extract_name_from_document(doc_data: Dict[str, Any]) -> Optional[str]:
    """Extract name field from document data."""
    name_fields = ['name', 'account_holder_name', 'customer_name', 'holder_name']
    for field in name_fields:
        if field in doc_data and doc_data[field]:
            return doc_data[field]
    return None


def extract_dob_from_document(doc_data: Dict[str, Any]) -> Optional[str]:
    """Extract DOB field from document data."""
    dob_fields = ['date_of_birth', 'dob', 'birth_date']
    for field in dob_fields:
        if field in doc_data and doc_data[field]:
            return doc_data[field]
    return None


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/verify")
async def verify(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    mode: str = Form(default="auto"),
    ensemble: bool = Form(default=False),
    runs: int = Form(default=3)
):
    """
    Verify two documents by comparing extracted data.
    Compares name and DOB across documents.
    """
    try:
        # Read both files
        content1 = await file1.read()
        content2 = await file2.read()
    except Exception as e:
        return JSONResponse({"error": f"Failed to read files: {str(e)}"}, status_code=500)

    try:
        # Extract data from document 1
        if ensemble:
            text1 = ensemble_tesseract_bytes(content1, runs=runs)
        else:
            text1 = choose_mode_and_scan(content1, mode)
        
        structured_data1 = extract_structured_data(text1)
        structured_data1["raw_text"] = text1
        
        # Extract data from document 2
        if ensemble:
            text2 = ensemble_tesseract_bytes(content2, runs=runs)
        else:
            text2 = choose_mode_and_scan(content2, mode)
        
        structured_data2 = extract_structured_data(text2)
        structured_data2["raw_text"] = text2
        
        # Verify documents using local functions
        name1 = extract_name_from_document(structured_data1)
        name2 = extract_name_from_document(structured_data2)
        dob1 = extract_dob_from_document(structured_data1)
        dob2 = extract_dob_from_document(structured_data2)
        
        # Calculate name similarity
        name_similarity = 0.0
        name_matched = False
        if name1 and name2:
            name_similarity = calculate_name_similarity(name1, name2)
            name_matched = name_similarity >= 0.8
        
        # Compare DOB
        dob_matched, dob_message = compare_dob(dob1, dob2)
        
        # Determine overall status
        status = "FAILED"
        overall_confidence = 0.0
        if name_matched and dob_matched:
            status = "VERIFIED"
            overall_confidence = (name_similarity * 100 + 100) / 2
        elif name_matched or dob_matched:
            status = "PARTIAL"
            overall_confidence = name_similarity * 100 if name_matched else 50.0
        else:
            status = "FAILED"
            overall_confidence = name_similarity * 100
        
        verification_result = {
            "status": status,
            "overall_confidence": round(overall_confidence, 2),
            "name_match": {
                "similarity": round(name_similarity * 100, 2),
                "matched": name_matched,
                "value1": name1,
                "value2": name2,
                "normalized1": normalize_name(name1) if name1 else None,
                "normalized2": normalize_name(name2) if name2 else None
            },
            "dob_match": {
                "matched": dob_matched,
                "value1": dob1,
                "value2": dob2,
                "message": dob_message
            },
            "documents": {
                "document1_type": structured_data1.get('document_type', 'Unknown'),
                "document2_type": structured_data2.get('document_type', 'Unknown')
            }
        }
        
        return {
            "document1": {
                "filename": file1.filename,
                "data": structured_data1
            },
            "document2": {
                "filename": file2.filename,
                "data": structured_data2
            },
            "verification": verification_result
        }
        
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"error": str(e), "traceback": tb}, status_code=500)
