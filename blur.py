import cv2 
  
# Load the Haar Cascade for face detection
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 

# Capture video from the default webcam (0 for the laptop webcam)
video_capture = cv2.VideoCapture(0) 

while True: 
    
    
    check, frame = video_capture.read() 
  
    # Convert the frame into grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
  
    # Detect multiple faces in the captured frame
    face = cascade.detectMultiScale(gray_image, scaleFactor=2.0, minNeighbors=4) 
  
    # Loop over the detected faces
    for x, y, w, h in face: 
  
       
        image = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3) 
  
        # Blur the face inside the rectangle
        image[y:y+h, x:x+w] = cv2.medianBlur(image[y:y+h, x:x+w], 35) 
  
   
    cv2.imshow('face blurred', frame) 
    key = cv2.waitKey(1) 
  
    # Break the loop if the 'q' key is pressed
    if key == ord('q'): 
        break
  

video_capture.release() 
cv2.destroyAllWindows()
