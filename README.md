# OCR Document Extraction & Verification System

A powerful OCR-based system for extracting structured data from Indian identity documents (Aadhaar, PAN, Bank Passbooks) with built-in document verification capabilities.

## 🌟 Features

### Document OCR Extraction
- ✅ **Aadhaar Card**: Name, Aadhaar Number, DOB, Address, Gender, Pincode
- ✅ **PAN Card**: Name, PAN Number, Father's Name, DOB
- ✅ **Bank Passbook**: Account Holder Name, Account Number, IFSC, MICR, Branch Name

### Document Verification
- ✅ **2-Document Comparison**: Compare name and DOB across documents
- ✅ **Fuzzy Name Matching**: Handles OCR errors with 80% similarity threshold
- ✅ **Multiple DOB Formats**: Supports DD/MM/YYYY, YYYY-MM-DD, and more
- ✅ **Confidence Scoring**: Returns verification confidence (0-100%)

### Advanced Features
- ✅ **Ensemble Mode**: Multi-pass OCR for higher accuracy
- ✅ **Auto Document Detection**: Automatically identifies document type
- ✅ **Image Preprocessing**: Enhances image quality for better OCR
- ✅ **Multiple Extraction Strategies**: 5+ fallback strategies per field

---

## 📁 Project Structure

```
ocr/
├── README.md                 # This file
├── test_images/              # Sample test images
│   ├── aadhar-dummy.png
│   ├── sbi-photo.jpg
│   └── WhatsApp Image *.jpeg
└── ocr_web/
    ├── backend/
    │   ├── main.py          # All backend code (FastAPI + OCR logic)
    │   ├── requirements.txt # Python dependencies
    │   └── __init__.py
    └── frontend/
        └── index.html       # Web interface
```

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.7+**
2. **Tesseract OCR** - Install from:
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - Mac: `brew install tesseract`

### Installation

1. **Clone the repository**
```bash
cd ocr
```

2. **Create virtual environment**
```bash
python -m venv .venv
```

3. **Activate virtual environment**
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

4. **Install dependencies**
```bash
cd ocr_web/backend
pip install -r requirements.txt
```

5. **Start the server**
```bash
python -m uvicorn main:app --reload --port 8000
```

6. **Open in browser**
```
http://127.0.0.1:8000
```

---

## 📖 Usage Guide

### Single Document OCR

1. Navigate to http://127.0.0.1:8000
2. Click **"Choose File"** and select a document image
3. Check **"Extract structured data"** checkbox
4. Click **"Upload & OCR"**
5. View extracted JSON data

**Example Output (Aadhaar):**
```json
{
  "document_type": "Aadhaar Card",
  "name": "Rajesh Kumar",
  "aadhaar_number": "123456789012",
  "date_of_birth": "15/08/1990",
  "gender": "MALE",
  "address": "123 Main Street, Delhi",
  "pincode": "110001"
}
```

### 2-Document Verification

1. Scroll to **"Document Verification"** section
2. Upload **Document 1** (e.g., PAN card)
3. Upload **Document 2** (e.g., Aadhaar card)
4. Click **"Verify Documents"**
5. View verification results with confidence score

**Example Output:**
```json
{
  "status": "VERIFIED",
  "overall_confidence": 95.5,
  "name_match": {
    "similarity": 91.0,
    "matched": true,
    "value1": "Rajesh Kumar",
    "value2": "Rajesh Kumar"
  },
  "dob_match": {
    "matched": true,
    "message": "DOB matches: 1990-08-15"
  }
}
```

---

## 🎯 Extraction Accuracy

| Document Type | Field | Accuracy | Strategies |
|--------------|-------|----------|-----------|
| **Aadhaar** | Name | ~80% | 5 fallback strategies |
| | Aadhaar Number | ~95% | Pattern matching |
| | DOB | ~85% | Multiple formats |
| | Address | ~75% | Line collection |
| **PAN** | Name | ~70% | 5 fallback strategies |
| | PAN Number | ~90% | Pattern matching |
| | DOB | ~70% | 4 fallback strategies |
| **Bank** | Account Holder | ~90% | 2 strategies + field combo |
| | Account Number | ~95% | Pattern matching |
| | IFSC | ~70% | Depends on image quality |

---

## 🔧 API Endpoints

### 1. GET `/`
Serves the frontend HTML interface.

### 2. POST `/upload`
Upload and extract data from a single document.

**Request:**
- `file`: Image file (multipart/form-data)
- `mode`: OCR mode (auto/fast/accurate) - optional
- `structured`: Extract structured data (true/false) - optional
- `ensemble`: Use ensemble mode (true/false) - optional

**Response:**
```json
{
  "filename": "aadhaar.jpg",
  "structured_data": {
    "document_type": "Aadhaar Card",
    "name": "Rajesh Kumar",
    ...
  }
}
```

### 3. POST `/verify`
Verify two documents by comparing name and DOB.

**Request:**
- `file1`: First document image
- `file2`: Second document image
- `mode`: OCR mode - optional
- `ensemble`: Use ensemble mode - optional

**Response:**
```json
{
  "document1": { ... },
  "document2": { ... },
  "verification": {
    "status": "VERIFIED",
    "overall_confidence": 95.5,
    "name_match": { ... },
    "dob_match": { ... }
  }
}
```

---

## 🛠️ Configuration

### OCR Modes

**Auto Mode (Default):**
- Automatically selects best OCR configuration
- Balanced speed and accuracy

**Fast Mode:**
- Quick processing
- Lower accuracy
- Use for: Bulk processing

**Accurate Mode:**
- Slower processing
- Higher accuracy
- Use for: Critical documents

**Ensemble Mode:**
- Multiple OCR passes (default: 3)
- Highest accuracy
- Slowest processing
- Use for: Maximum accuracy needed

### Verification Thresholds

**Name Matching:**
- Threshold: 80% similarity
- Uses fuzzy string matching (Levenshtein distance)
- Handles OCR errors and variations

**DOB Matching:**
- Exact match or year-only match
- Supports 10+ date formats
- Normalizes to YYYY-MM-DD

**Verification Status:**
- `VERIFIED`: Name ≥80% AND DOB matches
- `PARTIAL`: Name ≥80% OR DOB matches
- `FAILED`: Name <80% AND DOB doesn't match

---

## 📊 Extraction Strategies

### Aadhaar Name Extraction (5 Strategies)
1. Line before DOB
2. Explicit "Name:" label
3. Capitalized words in first few lines
4. Line before "Father's Name"
5. Largest all-caps block

### PAN Name Extraction (5 Strategies)
1. All-caps name (2-4 words)
2. "Name:" label
3. Text before PAN number
4. Line before "Father's Name"
5. Largest caps block

### Bank Account Holder (2 Strategies)
1. Labeled fields (Account Holder, Customer Name)
2. First + Middle + Last name combination

### DOB Extraction (4 Strategies)
1. Labeled DOB field
2. Date pattern near "Birth" keyword
3. Any date pattern (DD/MM/YYYY)
4. Year only (1900-2020)

---

## 🐛 Troubleshooting

### Issue: Low OCR Accuracy

**Solutions:**
1. Use high-resolution images (300+ DPI)
2. Ensure good lighting when photographing
3. Enable **Ensemble Mode** for better accuracy
4. Avoid blurry or low-quality images

### Issue: Name Not Extracted

**Solutions:**
1. Check if name is clearly visible in image
2. Try ensemble mode
3. Ensure name is in English (not regional language)
4. Check if document is properly oriented

### Issue: Verification Fails

**Solutions:**
1. Check if both documents have same name spelling
2. Verify DOB format is readable
3. Use ensemble mode for both documents
4. Check if OCR extracted correct data

### Issue: Server Won't Start

**Solutions:**
1. Verify Tesseract is installed: `tesseract --version`
2. Check Python version: `python --version` (3.7+)
3. Reinstall dependencies: `pip install -r requirements.txt`
4. Check port 8000 is not in use

---

## 📝 Best Practices

### For Best OCR Results:
1. ✅ Use **high-quality scans** (300+ DPI)
2. ✅ Ensure **good lighting** and contrast
3. ✅ Avoid **shadows and glare**
4. ✅ Keep camera **steady** (no blur)
5. ✅ Use **ensemble mode** for critical extractions

### For Document Verification:
1. ✅ Upload documents with **clear text**
2. ✅ Ensure **same name spelling** across documents
3. ✅ Use **consistent DOB format**
4. ✅ Enable **ensemble mode** for both documents

---

## 🔒 Security Notes

- This system processes documents **locally** - no data sent to external servers
- Images are **not stored** after processing
- All processing happens in **memory**
- No database or persistent storage

---

## 📦 Dependencies

```
fastapi          # Web framework
uvicorn          # ASGI server
python-multipart # File upload support
Pillow           # Image processing
opencv-python    # Advanced image processing
numpy            # Numerical operations
pytesseract      # Tesseract OCR wrapper
```

---

## 🚧 Limitations

1. **OCR Accuracy**: Depends on image quality
2. **Language Support**: Primarily English text
3. **Handwritten Text**: Not supported
4. **Stylized Fonts**: May reduce accuracy
5. **Document Variations**: Different layouts may affect extraction

---

## 🔮 Future Enhancements

- [ ] Support for more document types (Driving License, Voter ID)
- [ ] Multi-language support (Hindi, regional languages)
- [ ] Machine learning-based extraction
- [ ] Confidence scores for each extracted field
- [ ] Manual correction UI
- [ ] Batch processing support
- [ ] Address verification with postal database

---

## 📄 License

This project is provided as-is for educational and development purposes.

---

## 👨‍💻 Support

For issues or questions:
1. Check the **Troubleshooting** section above
2. Review the **Usage Guide**
3. Verify **Prerequisites** are installed correctly

---

## 🎉 Acknowledgments

- **Tesseract OCR** - Google's open-source OCR engine
- **FastAPI** - Modern Python web framework
- **OpenCV** - Computer vision library

---

**Made with ❤️ for accurate document processing**
