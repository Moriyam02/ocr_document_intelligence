import time
import pytesseract
import easyocr
from paddleocr import PaddleOCR
from PIL import Image

# Initialize EasyOCR and PaddleOCR instances (cached in memory)
easy_reader = easyocr.Reader(['en'], gpu=False)
paddle_reader = PaddleOCR(use_angle_cls=True, lang='en')

def run_tesseract(image_path: str) -> dict:
    start_time = time.time()
    image = Image.open(image_path)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    extracted_words, confidences, boxes = [], [], []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = float(data['conf'][i])
        if text and conf > 0:
            extracted_words.append(text)
            confidences.append(conf)
            boxes.append({
                "text": text,
                "confidence": round(conf, 2),
                "box": [data['left'][i], data['top'][i], data['width'][i], data['height'][i]]
            })
            
    avg_conf = (sum(confidences) / len(confidences)) / 100.0 if confidences else 0.0
    return {
        "engine_name": "Tesseract",
        "raw_text": " ".join(extracted_words),
        "confidence": round(avg_conf, 4),
        "duration_seconds": round(time.time() - start_time, 3),
        "boxes": boxes
    }

def run_easyocr(image_path: str) -> dict:
    start_time = time.time()
    results = easy_reader.readtext(image_path)
    extracted_words, confidences, boxes = [], [], []
    
    for bbox, text, conf in results:
        text_clean = text.strip()
        if text_clean:
            extracted_words.append(text_clean)
            confidences.append(conf)
            boxes.append({
                "text": text_clean,
                "confidence": round(float(conf), 2),
                "box": [[int(pt[0]), int(pt[1])] for pt in bbox]
            })
            
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return {
        "engine_name": "EasyOCR",
        "raw_text": " ".join(extracted_words),
        "confidence": round(avg_conf, 4),
        "duration_seconds": round(time.time() - start_time, 3),
        "boxes": boxes
    }

def run_paddleocr(image_path: str) -> dict:
    start_time = time.time()
    results = paddle_reader.ocr(image_path, cls=True)
    extracted_words, confidences, boxes = [], [], []
    
    if results and results[0]:
        for line in results[0]:
            bbox, (text, conf) = line
            text_clean = text.strip()
            if text_clean:
                extracted_words.append(text_clean)
                confidences.append(conf)
                boxes.append({
                    "text": text_clean,
                    "confidence": round(float(conf), 2),
                    "box": bbox
                })
                
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return {
        "engine_name": "PaddleOCR",
        "raw_text": " ".join(extracted_words),
        "confidence": round(avg_conf, 4),
        "duration_seconds": round(time.time() - start_time, 3),
        "boxes": boxes
    }

def run_all_ocr_engines(image_path: str) -> list[dict]:
    results = []
    for engine_func in [run_tesseract, run_easyocr, run_paddleocr]:
        try:
            results.append(engine_func(image_path))
        except Exception as e:
            results.append({
                "engine_name": engine_func.__name__.replace("run_", "").capitalize(),
                "raw_text": "", "confidence": 0.0, "duration_seconds": 0.0, "boxes": [], "error": str(e)
            })
    return results