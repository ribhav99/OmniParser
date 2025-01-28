from paddleocr import PaddleOCR

image_path = 'imgs/word.png'
paddle_ocr = PaddleOCR(
    lang='en',  # other lang also available
    use_angle_cls=False,
    use_gpu=False,  # using cuda will conflict with pytorch in the same process
    show_log=False,
    max_batch_size=1024,
    use_dilation=True,  # improves accuracy
    det_db_score_mode='slow',  # improves accuracy
    rec_batch_num=1024)

# results = paddle_ocr.ocr(image_path, cls=True)
result = paddle_ocr.ocr(image_path, cls=False)[0]
print(result)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# from utils import get_caption_model_processor
# caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence", device=device)
# model, processor = caption_model_processor['model'], caption_model_processor['processor']
# inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt").to(device=device)
# generated_ids = model.generate(input_ids=inputs["input_ids"],pixel_values=inputs["pixel_values"],max_new_tokens=100,num_beams=3, do_sample=False)