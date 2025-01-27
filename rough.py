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