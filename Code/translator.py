import cv2
import time
import mediapipe as mp
import numpy as np
import tensorflow as tf


#Load the model	
# ~ model = tf.keras.models.load_model("cnn_model_video.h5")
model = tf.keras.models.load_model("cnn_data_aug_model_video.h5")

#Label 
targets_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L"]


#Size
dim = (28, 28)


# Grabbing the Hand Model from Mediapipe and 
# Initializing the Model
mp_hands = mp.solutions.hands
hand_model = mp_hands.Hands(static_image_mode = True,
    min_detection_confidence=0.5
)

# Initializing the drawing utils for drawing the hand landmarks on image
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils

# (0) in VideoCapture is used to connect to your computer's default camera
capture = cv2.VideoCapture(0)

while capture.isOpened():

	xmax = 0
	xmin = 10000
	ymax = 0
	ymin = 10000

	# capture frame by frame
	ret, frame = capture.read()

	# resizing the frame for better view
	frame = cv2.resize(frame, (800, 600))

	# Converting the from BGR to RGB
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	results = hand_model.process(image)

	# Converting back the RGB image to BGR
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	cv2.imshow("Hand Landmarks", image)

	if results.multi_hand_landmarks:
		for hand_landmarks in results.multi_hand_landmarks:
			for k in range(21):     #Car y'a 21 points en tout 
				#Calcul des points maximum pour tracer le rectangle
				if hand_landmarks.landmark[k].x > xmax:
					xmax = hand_landmarks.landmark[k].x
				if hand_landmarks.landmark[k].y >ymax:
					ymax = hand_landmarks.landmark[k].y
				if hand_landmarks.landmark[k].x < xmin:
					xmin = hand_landmarks.landmark[k].x
				if hand_landmarks.landmark[k].y < ymin:
					ymin =hand_landmarks.landmark[k].y


			# Obtenir les dimensions de l'image
			(h, w) = image.shape[:2]

			#Position des points en pixels dans le référentiel de l'image avec le (0,0) en haut à gauche
			#Les valeurs sont arbitraires

			ymin = round(ymin*h)-30
			ymax = round(ymax*h)+10
			xmin = round(xmin*w)-30
			xmax = round(xmax*w)+30

			cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (255,0,0), 2)

			image2 = image[ymin:ymax,xmin:xmax]

			#Vector image
			img = np.zeros((w, h, 3), dtype = np.uint8)

			if (len(image2)!=0):
				#Pretreatment
				resized = cv2.resize(image2, dim, interpolation = cv2.INTER_AREA)
				gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
				reshaped = gray.reshape(-1, 28, 28, 1)
				
				#Prediction
				prediction = model.predict(reshaped);
				maximum = np.argmax(prediction)
				class_pred = targets_names[maximum]
				percent = round(prediction[0,maximum]*100)
				cv2.putText(image,class_pred+"  "+str(percent)+" % ",(xmin,ymax+25),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 0.8,color = (255, 255, 255),thickness=1)
			
			# Display the resulting image
			try:
				cv2.imshow("Hand Landmarks", image)
			except:
				print("Pas réussi a afficher")
                    
                
    # Enter key 'q' to break the loop
	if cv2.waitKey(5) & 0xFF == ord('q'):
		break

# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()

