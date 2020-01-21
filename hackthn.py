from flask import Flask , render_template,request
import pyrebase
import cv2
import numpy as np
from datetime import date
import face_recognition
import urllib.request
import os
from zipfile import ZipFile








#technex-2e504
Config = {
    "apiKey": "AIzaSyDAC0X4ANXg9uYcaP9Mfe6YgoQfsKiwvKA",
    "authDomain": "technex-2e504.firebaseapp.com",
    "databaseURL": "https://technex-2e504.firebaseio.com",
    "projectId": "technex-2e504",
    "storageBucket": "technex-2e504.appspot.com",
    "messagingSenderId": "57670336247",
    "appId": "1:57670336247:web:de64b4a76309dcc990ddf2",
    "measurementId": "G-PWD44WC9Y4"
  }

Today = date.today()
Today = str(Today)
list1=[]
known_face_names=[]
firbase = pyrebase.initialize_app(Config)
db = firbase.database()


app = Flask(__name__)


@app.route('/')

def index():
    return render_template('Login Form.html')

@app.route('/Attendance')
def main_pro():
    name = request.args.get('name')
    college_name = request.args.get('college_name')
    list1.append(name)
    list1.append(college_name)
    name=list1[0]
    college_name=list1[1]
   

    
    return render_template('attendance.html'  , name=name , college_name=college_name , date =Today)

@app.route('/completed')

def Report():
    
    student = db.child("Students").get()
    stu = student.val()
    for i in stu.values():
        known_face_names.append(i)  
    name = list1[0]
    college_name = list1[1]
    cap = cv2.VideoCapture(0)
    
    
    known_face_encodings = []
    present_list = []
    
    url = 'https://raw.githubusercontent.com/subrata3112/FaceRecognition/master/images.zip'
    urllib.request.urlretrieve(url, 'data/images.zip')

    with ZipFile('data/images.zip', 'r') as zip1:
        zip1.extractall('data/')
        

    
    
    for r,d,f in os.walk('/home/iiitk/Desktop/flask_practice/technex/data'):

        for filename in f:
            if 'jpg' in filename:
                src_image = face_recognition.load_image_file(filename)
                src_face_encoding = face_recognition.face_encodings(src_image)[0]
                known_face_encodings.append(src_face_encoding)



   

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:

        ret, frame = cap.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:

                matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.45)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    
                    if name not in present_list:
                        present_list.append(name)

                face_names.append(name)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()   
    url_data = "https://technex-2e504.firebaseio.com/"+str(list1[1])+".json"
    for i in range(1):
        db.child(list1[1]).push({Today:{"Name":present_list}})
       
    return render_template('final.html',name=name , college_name=college_name , url_data=url_data , date=Today)
    
 

    

if __name__=="__main__":
    app.run()



