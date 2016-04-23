import cv2,os
import numpy as np
from PIL import Image
import sys

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

print "Training"

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.face.createLBPHFaceRecognizer()

def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(1000)
    # return the images list and labels list
    return images, labels

# Path to the my Dataset
path = './yalefaces'
# Call the get_images_and_labels function and get the face images and the 
# corresponding labels
images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

# Perform the tranining
recognizer.train(images, np.array(labels))

counter = 0
video_capture = cv2.VideoCapture(0)
video_capture.set(3,640)
video_capture.set(4,480)
last_prediction = (0,0,0,0)
found_flag=False
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #claculate the face every three frames
    counter+=1
    counter=counter%3
    if counter == 0:
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
        )

        # Draw a rectangle around the faces
        if len(faces)!=0:
            found_flag = True
        else:
            found_flag = False
        for (x, y, w, h) in faces:
            nbr_predicted = recognizer.predict(gray[y:y+h,x:x+w])
            print nbr_predicted
            if (16 == nbr_predicted):
                last_prediction = (x,y,w,h)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                print "Size: ("+str(w)+","+str(h)+")"
    else:
        if found_flag:
            cv2.rectangle(frame,(last_prediction[0],last_prediction[1]),(last_prediction[0]+last_prediction[2],last_prediction[1]+last_prediction[3]),(0,255,0),2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
