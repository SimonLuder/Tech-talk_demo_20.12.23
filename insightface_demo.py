# simple demo of the insightface face recognition package 
# models can be downloaded from: https://github.com/deepinsight/insightface/tree/master/python-package

import cv2
import numpy as np
import insightface
from insightface.app.common import Face
from insightface.model_zoo import model_zoo

# detection model
DET_MODEL_PATH = 'models/buffalo_s/det_500m.onnx'

# recognition model
REC_MODEL_PATH = 'models/buffalo_s/w600k_mbf.onnx'

# Minimum similarity score to be verified
MIN_SIMILARITY = 0.4


def main():

    # load model weights
    det_model = model_zoo.get_model(DET_MODEL_PATH)
    rec_model = model_zoo.get_model(REC_MODEL_PATH)

    det_model.prepare(ctx_id=0, input_size=(640, 640), det_thres=0.5)

    # load reference image
    db_image = cv2.imread("reference_img.jpg")
    # detect face
    bboxes, face_keypoints = det_model.detect(db_image, max_num=0, metric='default') 
    # embed face
    for bbox, kps in zip(bboxes, face_keypoints):
        face = Face(bbox=bbox[:4], kps=kps, det_score=bbox[4])
        test_embedding = rec_model.get(db_image, face)

    
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        
        success, frame = cap.read()
        H, W, _ = frame.shape
        
        # detect faces
        bboxes, face_keypoints = det_model.detect(frame, max_num=0, metric='default') 

        for bbox, kps in zip(bboxes, face_keypoints):

            # embed detected faces
            face = Face(bbox=bbox[:4], kps=kps, det_score=bbox[4])
            live_embedding = rec_model.get(frame, face)

            # similarity
            cos_sim = np.dot(live_embedding, test_embedding)/(np.linalg.norm(live_embedding)*np.linalg.norm(test_embedding))

            if cos_sim >= MIN_SIMILARITY:
                color = (55, 255 ,55)
                message = "It's a match!"
            else:
                color = (55, 55 ,255)
                message = "I don't know you."
                
            # draw bbox
            frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1) 

            # draw similarity
            frame = cv2.putText(frame, f'{cos_sim:.3f}', (int(bbox[2]), int(bbox[3])), 
                                cv2.FONT_HERSHEY_SIMPLEX ,  1, color, 1, cv2.LINE_AA) 
            
            # draw verification message
            frame = cv2.putText(frame, message, (int(bbox[2]), int(bbox[3] * 0.9)), 
                                cv2.FONT_HERSHEY_SIMPLEX ,  1, color, 1, cv2.LINE_AA) 

            # draw face keypoints
            for kp in kps: 
                frame = cv2.circle(frame, (int(kp[0]), int(kp[1])), radius=1, color=color, thickness=-1)        


        cv2.imshow("Frame", frame)

        # close window with "q" key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":

    main()