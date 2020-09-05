
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2

prototxt_path = "deploy.prototxt.txt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
dnn_model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

print("Opening WebCam")
vs = VideoStream(src=0,framerate=30).start()
time.sleep(2.0)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=640,height=480)

	# Fulfiling the required input frame condition of dnn model
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	dnn_model.setInput(blob)
	detections = np.squeeze(dnn_model.forward())

	for i in range(0, detections.shape[0]):
		confidence = detections[i, 2]

		if confidence < 0.45:
			continue
		# In detection matrix ,at index =3 to 7 there are values present of startx, starty, endx, endy
		box = detections[i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype(np.int)

		probability = "{:.2f}%".format(confidence * 100)

		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(255, 255, 0), 3)

		cv2.putText(frame, probability, (startX, y),
			cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.45, (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("Real Time Output", frame)
	stopkey = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if stopkey == ord("q"):
		print("Closing WebCam")
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
print("Program is completed")