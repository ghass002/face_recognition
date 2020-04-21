import face_recognition
import os 
import cv2

KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.6 #more matches but risk to have errors set tolerance to be high, if you want a match only if its for sure, set it low
# for the detected frame
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
# you can use hog insted of cnn if you only have cpu and it is too slow
MODEL = "cnn"

video = cv2.VideoCapture(0)

print("loading known faces")

known_faces = []
known_names = []


for name in os.listdir(KNOWN_FACES_DIR):
	for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
		image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
		encoding = face_recognition.face_encodings(image)[0] 
		known_faces.append(encoding)
		known_names.append(name)

print("processing unknown faces")

while True:
	
	ret, image = video.read()
	#face detections
	locations = face_recognition.face_locations(image, model=MODEL)
	encodings = face_recognition.face_encodings(image, locations)
	

	for face_encoding, face_location in zip(encodings,locations):
		results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
		match = None
		if True in results:
			match = known_names[results.index(True)]
			print(f"Match found:{match}")

			# Draw the rectangle arround the face, we only need top left and bottom right corner to draw a rect
			top_left = (face_location[3], face_location[0])
			bottom_right = (face_location[1], face_location[2])
			color = [0,255,0]
			cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
			# draw rectangle to write the identity
			top_left = (face_location[3], face_location[2])
			bottom_right = (face_location[1], face_location[2]+22)
			cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
			cv2.putText(image, match,(face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THICKNESS)

	cv2.imshow(filename, image)
	
	if cv2.waitKey(1) & 0xFF == ord("q"): 
		break

	