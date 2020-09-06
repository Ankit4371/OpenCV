import cv2
###############################
framewidth = 640
frameheight = 480
numPlatCascade = cv2.CascadeClassifier('model')
minArea =500
color = (255,0,0)
###############################
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,20)
count = 0
while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = numPlatCascade.detectMultiScale(imgGray, 1.1, 4)
    for (x, y, w, h) in numberPlates:
        area = w*h
        if area> minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img,"Number Plate",(x,y-4),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,color,2)
            imgROI = img[y:y+h,x:x+w]
            cv2.imshow("Output",imgROI)
    cv2.imshow("Result",img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Number Plates/Detected_"+str(count)+".jpg" , imgROI)
        cv2.rectangle(img,(2,100),(640,300),(0,255,0),cv2.FILLED)
        cv2.putText(img,"Scan Saved",(150,265),cv2.FONT_HERSHEY_TRIPLEX,2,(0,0,255),2)
        cv2.imshow("Result",img)
        cv2.waitKey(500)
        count+=1
    stop = cv2.waitKey(30) & 0xff
    if stop == 27:
        break

cv2.destroyAllWindows()
