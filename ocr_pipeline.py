class OCRManager:
    def __init__(self):
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(
            lang='en',
            enable_mkldnn=False,
            text_detection_model_name='PP-OCRv5_mobile_det',
            text_recognition_model_name='en_PP-OCRv5_mobile_rec',
            text_det_limit_side_len=960,
            text_det_limit_type='max',
            use_doc_unwarping=False,
        )

    def _resize_for_ocr(self, image_path, max_side=1600):
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        h, w = img.shape[:2]
        if max(h, w) <= max_side:
            return image_path
        scale = max_side / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_path = image_path.replace(".png", "_ocr.png")
        cv2.imwrite(resized_path, resized)
        return resized_path

    def extract_text(self, image_path):
        image_path = self._resize_for_ocr(image_path)
        result = self.ocr.ocr(image_path)
        full_text = []
        extracted_data = []
        if not result:
            return "", []
        for page_res in result:
            if page_res is None:
                continue
            texts = []
            scores = []
            if hasattr(page_res, 'keys'):
                texts = page_res.get('rec_texts', page_res.get('text', page_res.get('label', [])))
                scores = page_res.get('rec_scores', page_res.get('score', page_res.get('confidence', [])))
                if not texts and not isinstance(page_res, list):
                    texts = getattr(page_res, 'rec_texts', getattr(page_res, 'text', []))
            elif isinstance(page_res, list):
                for line in page_res:
                    try:
                        if isinstance(line, list) and len(line) >= 2:
                            text = line[1][0]
                            score = line[1][1]
                            full_text.append(str(text).strip())
                            extracted_data.append({
                                "text": str(text).strip(),
                                "confidence": float(score),
                                "coords": line[0]
                            })
                    except: continue
                continue
            if texts and isinstance(texts, list):
                if not scores or len(scores) != len(texts):
                    scores = [1.0] * len(texts)
                for t, s in zip(texts, scores):
                    if t and str(t).strip():
                        full_text.append(str(t).strip())
                        extracted_data.append({
                            "text": str(t).strip(),
                            "confidence": float(s) if s else 1.0,
                            "coords": []
                        })
        final_string = " ".join(full_text)
        return final_string, extracted_data

if __name__ == "__main__":
    manager = OCRManager()
