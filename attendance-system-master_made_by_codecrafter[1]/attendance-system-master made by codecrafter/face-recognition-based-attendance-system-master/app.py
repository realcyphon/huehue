# import cv2
# import os
# from flask import Flask, request, render_template
# from datetime import date
# from datetime import datetime
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# import pandas as pd
# import joblib

# # Defining Flask App
# app = Flask(__name__)

# nimgs = 10

# # Saving Date today in 2 different formats
# datetoday = date.today().strftime("%m_%d_%y")
# datetoday2 = date.today().strftime("%d-%B-%Y")


# # Initializing VideoCapture object to access WebCam
# face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# # If these directories don't exist, create them
# if not os.path.isdir('Attendance'):
#     os.makedirs('Attendance')
# if not os.path.isdir('static'):
#     os.makedirs('static')
# if not os.path.isdir('static/faces'):
#     os.makedirs('static/faces')
# if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
#     with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
#         f.write('Name,Roll,Time')


# # get a number of total registered users
# def totalreg():
#     return len(os.listdir('static/faces'))


# # extract the face from an image
# def extract_faces(img):
#     try:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
#         return face_points
#     except:
#         return []


# # Identify face using ML model
# def identify_face(facearray):
#     model = joblib.load('static/face_recognition_model.pkl')
#     return model.predict(facearray)


# # A function which trains the model on all the faces available in faces folder
# def train_model():
#     faces = []
#     labels = []
#     userlist = os.listdir('static/faces')
#     for user in userlist:
#         for imgname in os.listdir(f'static/faces/{user}'):
#             img = cv2.imread(f'static/faces/{user}/{imgname}')
#             resized_face = cv2.resize(img, (50, 50))
#             faces.append(resized_face.ravel())
#             labels.append(user)
#     faces = np.array(faces)
#     knn = KNeighborsClassifier(n_neighbors=5)
#     knn.fit(faces, labels)
#     joblib.dump(knn, 'static/face_recognition_model.pkl')


# # Extract info from today's attendance file in attendance folder
# def extract_attendance():
#     df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
#     names = df['Name']
#     rolls = df['Roll']
#     times = df['Time']
#     l = len(df)
#     return names, rolls, times, l


# # Add Attendance of a specific user
# def add_attendance(name):
#     username = name.split('_')[0]
#     userid = name.split('_')[1]
#     current_time = datetime.now().strftime("%H:%M:%S")

#     df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
#     if int(userid) not in list(df['Roll']):
#         with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
#             f.write(f'\n{username},{userid},{current_time}')


# ## A function to get names and rol numbers of all users
# def getallusers():
#     userlist = os.listdir('static/faces')
#     names = []
#     rolls = []
#     l = len(userlist)

#     for i in userlist:
#         name, roll = i.split('_')
#         names.append(name)
#         rolls.append(roll)

#     return userlist, names, rolls, l


# ## A function to delete a user folder 
# def deletefolder(duser):
#     pics = os.listdir(duser)
#     for i in pics:
#         os.remove(duser+'/'+i)
#     os.rmdir(duser)




# ################## ROUTING FUNCTIONS #########################

# # Our main page
# @app.route('/')
# def home():
#     names, rolls, times, l = extract_attendance()
#     return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# ## List users page
# @app.route('/listusers')
# def listusers():
#     userlist, names, rolls, l = getallusers()
#     return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# ## Delete functionality
# @app.route('/deleteuser', methods=['GET'])
# def deleteuser():
#     duser = request.args.get('user')
#     deletefolder('static/faces/'+duser)

#     ## if all the face are deleted, delete the trained file...
#     if os.listdir('static/faces/')==[]:
#         os.remove('static/face_recognition_model.pkl')
    
#     try:
#         train_model()
#     except:
#         pass

#     userlist, names, rolls, l = getallusers()
#     return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# # Our main Face Recognition functionality. 
# # This function will run when we click on Take Attendance Button.
# @app.route('/start', methods=['GET'])
# def start():
#     names, rolls, times, l = extract_attendance()

#     if 'face_recognition_model.pkl' not in os.listdir('static'):
#         return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

#     ret = True
#     cap = cv2.VideoCapture(0)
#     while ret:
#         ret, frame = cap.read()
#         if len(extract_faces(frame)) > 0:
#             (x, y, w, h) = extract_faces(frame)[0]
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
#             cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
#             face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
#             identified_person = identify_face(face.reshape(1, -1))[0]
#             add_attendance(identified_person)
#             cv2.putText(frame, f'{identified_person}', (x+5, y-5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#         cv2.imshow('Attendance', frame)
#         if cv2.waitKey(1) == 27:
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     names, rolls, times, l = extract_attendance()
#     return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# # A function to add a new user.
# # This function will run when we add a new user.
# @app.route('/add', methods=['GET', 'POST'])
# def add():
#     newusername = request.form['newusername']
#     newuserid = request.form['newuserid']
#     userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
#     if not os.path.isdir(userimagefolder):
#         os.makedirs(userimagefolder)
#     i, j = 0, 0
#     cap = cv2.VideoCapture(0)
#     while 1:
#         _, frame = cap.read()
#         faces = extract_faces(frame)
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
#             cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
#             if j % 5 == 0:
#                 name = newusername+'_'+str(i)+'.jpg'
#                 cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
#                 i += 1
#             j += 1
#         if j == nimgs*5:
#             break
#         cv2.imshow('Adding new User', frame)
#         if cv2.waitKey(1) == 27:
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     print('Training Model')
#     train_model()
#     names, rolls, times, l = extract_attendance()
#     return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# # Our main function which runs the Flask App
# if __name__ == '__main__':
#     app.run(debug=True)


# Import necessary libraries
import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# Initialize Flask App
app = Flask(__name__)

# Set number of images to capture for new user
NUM_IMAGES_TO_CAPTURE = 10

# File paths
CASCADE_FILE_PATH = 'haarcascade_frontalface_default.xml'
STATIC_FACES_FOLDER = 'static/faces'
ATTENDANCE_FOLDER = 'Attendance'
MODEL_FILE_PATH = 'static/face_recognition_model.pkl'

# Initialize face detector
face_detector = cv2.CascadeClassifier(CASCADE_FILE_PATH)

# Ensure required directories exist
for folder in [ATTENDANCE_FOLDER, STATIC_FACES_FOLDER]:
    if not os.path.isdir(folder):
        os.makedirs(ATTENDANCE_FOLDER)

# Set today's date
TODAY_DATE = date.today().strftime("%m_%d_%y")
TODAY_DATE_FORMATTED = date.today().strftime("%d-%B-%Y")

# Function to extract faces from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except Exception as e:
        print("Error extracting faces:", e)
        return []

# Function to identify a face using the ML model
def identify_face(facearray):
    try:
        model = joblib.load(MODEL_FILE_PATH)
        return model.predict(facearray)
    except Exception as e:
        print("Error identifying face:", e)
        return []

# Function to train the ML model
def train_model():
    try:
        faces = []
        labels = []
        for user in os.listdir(STATIC_FACES_FOLDER):
            for imgname in os.listdir(os.path.join(STATIC_FACES_FOLDER, user)):
                img = cv2.imread(os.path.join(STATIC_FACES_FOLDER, user, imgname))
                resized_face = cv2.resize(img, (50, 50))
                faces.append(resized_face.ravel())
                labels.append(user)
        faces = np.array(faces)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces, labels)
        joblib.dump(knn, MODEL_FILE_PATH)
    except Exception as e:
        print("Error training model:", e)

# Function to extract attendance information
def extract_attendance():
    try:
        df = pd.read_csv(os.path.join(ATTENDANCE_FOLDER, f'Attendance-{TODAY_DATE}.csv'))
        return df['Name'], df['Roll'], df['Time'], len(df)
    except Exception as e:
        print("Error extracting attendance:", e)
        return [], [], [], 0

# Function to add attendance for a specific user
def add_attendance(name):
    try:
        username, userid = name.split('_')
        current_time = datetime.now().strftime("%H:%M:%S")
        df = pd.read_csv(os.path.join(ATTENDANCE_FOLDER, f'Attendance-{TODAY_DATE}.csv'))
        if int(userid) not in df['Roll']:
            with open(os.path.join(ATTENDANCE_FOLDER, f'Attendance-{TODAY_DATE}.csv'), 'a') as f:
                f.write(f'\n{username},{userid},{current_time}')
    except Exception as e:
        print("Error adding attendance:", e)

# Function to get all users
def get_all_users():
    try:
        users = os.listdir(STATIC_FACES_FOLDER)
        names = [user.split('_')[0] for user in users]
        rolls = [user.split('_')[1] for user in users]
        return users, names, rolls, len(users)
    except Exception as e:
        print("Error getting all users:", e)
        return [], [], [], 0

# Function to delete a user's folder
def delete_user_folder(user_folder):
    try:
        user_path = os.path.join(STATIC_FACES_FOLDER, user_folder)
        for file in os.listdir(user_path):
            os.remove(os.path.join(user_path, file))
        os.rmdir(user_path)
        if not os.listdir(STATIC_FACES_FOLDER):
            os.remove(MODEL_FILE_PATH)
        train_model()
    except Exception as e:
        print("Error deleting user folder:", e)

# Route for home page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=len(os.listdir(STATIC_FACES_FOLDER)), datetoday2=TODAY_DATE_FORMATTED)

# Route for listing users
@app.route('/listusers')
def list_users():
    userlist, names, rolls, l = get_all_users()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=len(os.listdir(STATIC_FACES_FOLDER)), datetoday2=TODAY_DATE_FORMATTED)

# Route for deleting a user
@app.route('/deleteuser', methods=['GET'])
def delete_user():
    user_to_delete = request.args.get('user')
    delete_user_folder(user_to_delete)
    userlist, names, rolls, l = get_all_users()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=len(os.listdir(STATIC_FACES_FOLDER)), datetoday2=TODAY_DATE_FORMATTED)

# Route for starting face recognition
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if not os.path.exists(MODEL_FILE_PATH):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=len(os.listdir(STATIC_FACES_FOLDER)), datetoday2=TODAY_DATE_FORMATTED, mess='There is no trained model available. Please add faces to continue.')

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            faces = extract_faces(frame)
            if faces:
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
                cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]
                add_attendance(identified_person)
                cv2.putText(frame, f'{identified_person}', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) == 27:
                break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=len(os.listdir(STATIC_FACES_FOLDER)), datetoday2=TODAY_DATE_FORMATTED)

# Route for adding a new user
@app.route('/add', methods=['POST'])
def add_user():
    new_user_name = request.form['newusername']
    new_user_id = request.form['newuserid']
    user_image_folder = os.path.join(STATIC_FACES_FOLDER, f'{new_user_name}_{new_user_id}')
    if not os.path.isdir(user_image_folder):
        os.makedirs(user_image_folder)
    num_images_captured, count = 0, 0
    cap = cv2.VideoCapture(0)
    while count < NUM_IMAGES_TO_CAPTURE * 5:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            num_images_captured += 1
            if num_images_captured % 5 == 0:
                image_name = f'{new_user_name}_{count}.jpg'
                cv2.imwrite(os.path.join(user_image_folder, image_name), frame[y:y+h, x:x+w])
                count += 1
        cv2.putText(frame, f'Images Captured: {num_images_captured}/{NUM_IMAGES_TO_CAPTURE}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=len(os.listdir(STATIC_FACES_FOLDER)), datetoday2=TODAY_DATE_FORMATTED)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

