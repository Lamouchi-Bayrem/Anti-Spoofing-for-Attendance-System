import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from imutils.video import VideoStream
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import dlib
import face_recognition

# Initialize paths and models
KNOWN_FACES_DIR = 'known_faces'
ATTENDANCE_LOG = 'attendance.csv'
SPOOF_MODEL_PATH = 'anti_spoofing_model.h5'  # Pre-trained anti-spoofing model

# Load anti-spoofing model
spoof_model = load_model(SPOOF_MODEL_PATH)

# Initialize face detector (using dlib for better accuracy)
detector = dlib.get_frontal_face_detector()

# Load known faces
known_face_encodings = []
known_face_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)

# Initialize attendance log if not exists
if not os.path.exists(ATTENDANCE_LOG):
    pd.DataFrame(columns=['Name', 'Date', 'Time', 'Status', 'Confidence', 'Spoof_Score']).to_csv(ATTENDANCE_LOG, index=False)

def detect_spoof(frame, face_box):
    (x, y, w, h) = face_box
    face = frame[y:y+h, x:x+w]
    
    try:
        # Preprocess face for spoof detection
        face = cv2.resize(face, (160, 160))
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        
        # Predict real vs spoof
        preds = spoof_model.predict(face)[0]
        spoof_score = preds[1]  # Probability of being spoofed
        return spoof_score
    except:
        return 0.5  # Default score if processing fails

def mark_attendance(name, status, confidence, spoof_score):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    
    # Check if this is a valid check-in/out (not a spoof attempt)
    if spoof_score < 0.5:  # Threshold can be adjusted
        new_entry = pd.DataFrame([[name, date, time, status, confidence, spoof_score]],
                                columns=['Name', 'Date', 'Time', 'Status', 'Confidence', 'Spoof_Score'])
        
        # Append to log
        df = pd.read_csv(ATTENDANCE_LOG)
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_LOG, index=False)
        return True
    return False

def main():
    vs = VideoStream(src=0).start()
    
    while True:
        frame = vs.read()
        if frame is None:
            continue
            
        # Convert to RGB (for face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        
        # Convert dlib rectangles to face_recognition format
        face_locations = [(face.top(), face.right(), face.bottom(), face.left()) for face in faces]
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"
            confidence = 0
            spoof_score = 1.0  # Default to spoofed
            
            # Calculate face distance
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
                
                # Check for spoofing
                spoof_score = detect_spoof(frame, (left, top, right-left, bottom-top))
                
                # Draw rectangle and label
                color = (0, 255, 0) if spoof_score < 0.5 else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Display label
                label = f"{name} ({confidence:.2f}) {'REAL' if spoof_score < 0.5 else 'SPOOF'}"
                cv2.putText(frame, label, (left, top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Check if we should mark attendance
                if spoof_score < 0.5 and confidence > 0.6:
                    # Determine if this is check-in or check-out based on last status
                    df = pd.read_csv(ATTENDANCE_LOG)
                    user_logs = df[df['Name'] == name]
                    
                    if len(user_logs) == 0:
                        status = 'IN'
                    else:
                        last_status = user_logs.iloc[-1]['Status']
                        status = 'OUT' if last_status == 'IN' else 'IN'
                    
                    marked = mark_attendance(name, status, confidence, spoof_score)
                    if marked:
                        print(f"Attendance marked: {name} {status}")
        
        # Display the resulting image
        cv2.imshow('Face Recognition with Anti-Spoofing', frame)
        
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()
