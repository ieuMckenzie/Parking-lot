You want to cd into the my_model folder

Then you want to run python yolo_detect.py --model my_model.pt --source usb0 
This allows you to use your webcam as source
With --source you can use any image and video file you want as long as its in the my model folder

Example command for running with 2 cameras with frame skipping and more OCR workers:
python yolo_detect.py --model my_model.pt --source usb0,usb1 --skip-frames 3 --ocr-workers 4