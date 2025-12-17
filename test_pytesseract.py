import os
import pytesseract
from PIL import Image

def test_pytesseract_integration():
    print("🔍 Testing Pytesseract Library Integration")
    
    # Mirroring the logic from image2excel.py
    if os.name == 'nt':
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.join(os.getenv('LOCALAPPDATA', ''), r"Tesseract-OCR\tesseract.exe")
        ]
        tesseract_path = None
        for path in possible_paths:
            if os.path.exists(path):
                tesseract_path = path
                break
        
        if tesseract_path:
            print(f"✅ Logic selected path: {tesseract_path}")
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
            tessdata_dir = os.path.dirname(tesseract_path)
            os.environ["TESSDATA_PREFIX"] = os.path.join(tessdata_dir, "tessdata")
            print(f"ℹ️ Set TESSDATA_PREFIX: {os.environ['TESSDATA_PREFIX']}")
        else:
            print("❌ Logic did NOT find Tesseract path")
    
    # Create a dummy image to test OCR
    img = Image.new('RGB', (100, 30), color = (255, 255, 255))
    
    try:
        print("⏳ Attempting image_to_data...")
        # config matching image2excel.py
        config = "--psm 6 -l fra+eng --oem 3"
        data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
        print("✅ Pytesseract execution successful!")
        print(f"Data keys: {list(data.keys())}")
    except Exception as e:
        print(f"❌ Pytesseract failed: {e}")
        # Print the actual tesseract_cmd being used
        print(f"ℹ️ Current tesseract_cmd: {pytesseract.pytesseract.tesseract_cmd}")

if __name__ == "__main__":
    test_pytesseract_integration()
