import numpy as np
import cv2

prototxt_path = ""
model_path = ""
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

image = cv2.imread(" ")
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
model.setInput(blob)
detections = np.squeeze(model.forward())

for i in range(0, detections.shape[0]):
 
    confidence = detections[i, 2]
  
    if confidence > 0.5:
       
        box = detections[i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype(np.int)
        
        probability = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (255, 255, 255), 2)
        cv2.putText(image, probability, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.imwrite("detected image",image)
print("Done")
