import os
import uuid
import logging
import time
import zipfile
import multiprocessing
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from typing import List
import onnxruntime as ort

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration
MODEL_PATH = '/home/yash/Desktop/cctv-face-detection/cctv-face-dection-model.onnx'
TEMP_DIR = "/tmp/video_processing"
os.makedirs(TEMP_DIR, exist_ok=True)

# ONNX Runtime configuration
providers = ['CPUExecutionProvider']
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = multiprocessing.cpu_count()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Initialize ONNX model
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    ort_session = ort.InferenceSession(MODEL_PATH, 
                                      sess_options=session_options, 
                                      providers=providers)
    
    # Get model metadata
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    output_name = ort_session.get_outputs()[0].name
    
    logger.info(f"Model loaded | Input: {input_name} {input_shape} | Output: {output_name}")
    
except Exception as e:
    logger.error(f"Model initialization failed: {str(e)}")
    raise RuntimeError("Could not initialize model")

def letterbox(img, new_shape=640, color=(114, 114, 114), auto=False, scaleup=True):
    """Resize image and maintain aspect ratio"""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2  # Divide padding into 2 sides
    dh /= 2

    # Resize and pad
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                            cv2.BORDER_CONSTANT, value=color)
    return img, (r, (dw, dh))

def non_max_suppression(predictions, conf_thres=0.25, iou_thres=0.45):
    """Non-Maximum Suppression (NMS) on inference results"""
    # Validate input
    if len(predictions.shape) != 3:
        predictions = np.expand_dims(predictions, axis=0)
        
    # Filter by confidence
    xc = predictions[..., 4] > conf_thres
    output = [np.zeros((0, 6))] * predictions.shape[0]
    
    for xi, x in enumerate(predictions):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = x[:, :4].copy()
        box[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        box[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        box[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        box[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

        # Detections matrix nx6 (xyxy, conf, cls)
        conf = np.max(x[:, 5:], axis=1, keepdims=True)
        j = np.argmax(x[:, 5:], axis=1).reshape(conf.shape)
        x = np.concatenate((box, conf, j.astype(float)), axis=1)
        x = x[x[:, 4] > conf_thres]

        # Check shape
        n = x.shape[0]
        if not n:
            continue

        # NMS
        boxes = x[:, :4]
        scores = x[:, 4]
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
        
        if indices.size > 0:
            output[xi] = x[indices.flatten()]
            
    return output

def process_video(input_path: str, output_path: str):
    """Optimized video processing pipeline"""
    try:
        logger.info(f"Processing: {os.path.basename(input_path)}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing parameters
        FRAME_SKIP = 3  # Process 1 of every 4 frames
        MODEL_INPUT_SIZE = 640
        frame_count = 0
        last_detections = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            process_frame = frame_count % (FRAME_SKIP + 1) == 0

            if process_frame:
                # Preprocess with letterboxing
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized, (ratio, (dw, dh)) = letterbox(img, MODEL_INPUT_SIZE)
                
                # Convert to tensor
                blob = cv2.dnn.blobFromImage(
                    resized, 
                    1/255.0, 
                    (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), 
                    swapRB=False,
                    crop=False
                )
                
                # Run inference
                outputs = ort_session.run([output_name], {input_name: blob})[0]
                
                # Process detections
                detections = non_max_suppression(outputs, conf_thres=0.5)[0]
                last_detections = detections

            # Draw detections
            if last_detections is not None and len(last_detections) > 0:
                for det in last_detections:
                    # Rescale coordinates
                    x1 = int((det[0] - dw) / ratio)
                    y1 = int((det[1] - dh) / ratio)
                    x2 = int((det[2] - dw) / ratio)
                    y2 = int((det[3] - dh) / ratio)
                    conf = det[4]
                    
                    # Clamp coordinates
                    x1 = max(0, min(x1, width))
                    y1 = max(0, min(y1, height))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f"Face {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2
                    )

            out.write(frame)

        logger.info(f"Completed: {os.path.basename(input_path)}")
        return output_path

    except Exception as e:
        logger.error(f"Error processing {input_path}: {str(e)}")
        raise
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals() and out.isOpened():
            out.release()
        if os.path.exists(input_path):
            os.remove(input_path)

def process_wrapper(args):
    return process_video(*args)

@app.post("/process-videos")
async def handle_multiple_videos(files: List[UploadFile]):
    start= time.time()
    """Process multiple videos with parallel execution"""
    if not files:
        raise HTTPException(400, "No files provided")
    
    batch_id = uuid.uuid4().hex
    temp_files = []
    processing_args = []

    try:
        # Prepare files
        for file in files:
            if not file.content_type.startswith("video/"):
                continue
            
            file_id = uuid.uuid4().hex
            input_path = os.path.join(TEMP_DIR, f"{batch_id}_{file_id}_input.mp4")
            output_path = os.path.join(TEMP_DIR, f"{batch_id}_{file_id}_output.mp4")
            
            content = await file.read()
            with open(input_path, "wb") as f:
                f.write(content)
            
            temp_files.extend([input_path, output_path])
            processing_args.append((input_path, output_path))

        # Process in parallel
        pool_size = min(4, multiprocessing.cpu_count())
        with multiprocessing.Pool(pool_size) as pool:
            results = pool.map(process_wrapper, processing_args)

        # Create zip archive
        zip_path = os.path.join(TEMP_DIR, f"{batch_id}_results.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for result in results:
                if os.path.exists(result):
                    zipf.write(result, os.path.basename(result))
        print("time_taken",time.time()-start)
        return StreamingResponse(
            open(zip_path, "rb"),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=processed_videos.zip"}
        )

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        for path in temp_files:
            if os.path.exists(path):
                os.remove(path)
        raise HTTPException(500, f"Processing failed: {str(e)}")