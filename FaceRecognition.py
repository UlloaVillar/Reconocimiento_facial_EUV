
from PIL import Image
from mtcnn.mtcnn import MTCNN
from numpy import asarray
import numpy
from server import Streaming

import os
import numpy as np

# Embeddings
from numpy import expand_dims
from keras.models import load_model

# SVC Classifier

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

# For opencv
import cv2
import time


class FaceRecognition:
    def __init__(self):
        self.streaming = Streaming()

    def extract_face(self, image_path):
        # load an  image. Image format from PIL library
        image = Image.open(image_path)
        # Convert to RGB
        image = image.convert('RGB')
        # Converting image to an array of pixels. asarray function to convert image to array of pixels
        # from numpy
        image_array = asarray(image)

        # Using model of Iván Paz Centeno from ipaz/mtcnn project
        face_detector = MTCNN()

        # Getting the faces from the array of pixels
        faces = face_detector.detect_faces(image_array)

        # Detect_faces locates faces returning a reference to the bounding box of the faces
        # Detect_faces returns the botton left corner, the width and the height

        # Create the bounding box
        x1, y1, width, height = faces[0]['box']
        x1= abs(x1)
        y1= abs(y1)
        x2, y2 = x1 + width, y1 + height

        # lets create a new image with only the face
        face_detected = image_array[y1:y2, x1:x2]

        # Lets resize the images to be introduced in the Convulotional Network
        # lets change array to pixel to resize, and then lets
        # change again to array

        face_detected = Image.fromarray(face_detected)
        face_detected = face_detected.resize((160, 160))
        face_detected = asarray(face_detected)
        return face_detected

    def get_faces_from_directory(self, directory):
        faces_detected = []
        i = 0
        images_inside_directory = os.listdir(directory)
        for im in images_inside_directory:
            path = directory + '/' + im
            face = self.extract_face(path)
            faces_detected.append(face)
        return faces_detected

    def faces_of_identities(self, directory):
        ids_db = os.listdir(directory)
        X = []
        Y = []
        for c in ids_db:
            directory3 = directory + '/' + c
            # Function to get all images on the directory and return them
            faces_detected = self.get_faces_from_directory(directory3)
            # labeling each face, with a number and the name of the face owner
            labels = [c for im in range(len(faces_detected))] # space for every photo for the actual celebrity
            print('>loaded %d examples for class: %s' % (len(faces_detected), c))
            X.extend(faces_detected)
            Y.extend(labels)
        return X, Y

    def get_faces(self, directory):
        Xids_db = []
        Yids_db = []
        Xval = []
        Yval = []
        i = 0
        folders = os.listdir(directory)

        for f in folders:
            directory2 = directory + f
            if i == 1:
                Xval, Yval = self.faces_of_identities(directory2)
            if i == 0:
                Xids_db, Yids_db = self.faces_of_identities(directory2)
            i += 1

        return Xids_db, Yids_db, Xval, Yval

    def embed_face(self, model, face):
        # we need to standardize and to scale
        face = face.astype('float32')
        mean, std = face.mean(), face.std()
        face = (face - mean) / std  # data standarization
        samples = expand_dims(face, axis=0)
        yhat = model.predict(samples)
        return yhat[0]

    def embed_set(self, set, model):

        embedding_model = load_model(model)
        embeddings = []
        for X in set:
            face_embedded = self.embed_face(embedding_model, X)
            embeddings.append(face_embedded)
        return embeddings


    def extract_face_webacm(self, frame, face_detector):

        # Converting image to an array of pixels. asarray function to convert image to array of pixels
        frame = asarray(frame)

        # Getting the faces from the array of pixels.
        # MTCNN model
        faces = face_detector.detect_faces(frame)

        # Detect_faces locates faces returning a reference to the bounding box of the faces
        # Detect_faces returns the botton left corner, the width and the height

        if len(faces) >= 1:
            # If face found
            # Create the bounding box
            x1, y1, width, height = faces[0]['box']
            x1 = abs(x1)
            y1 = abs(y1)
            x2, y2 = x1 + width, y1 + height

            # image with only the face
            face_detected = frame[y1:y2, x1:x2]

            # Lets resize the images to be introduced in the Convolutional Network

            # lets change array to pixel to resize, and then lets
            # change again to array

            face_detected = Image.fromarray(face_detected)
            face_detected = face_detected.resize((160, 160))
            face_detected = asarray(face_detected)
            return face_detected
        else:
            return frame

    def train_classifier_with(self, directory):
        # Extract faces
        Xids_db, Yids_db, Xtest, Ytest = self.get_faces(directory)

        print("All faces have been extracted from the dataset")

        # Get embeddings from the faces
        Xids_db = self.embed_set(Xids_db, 'facenet_keras.h5')
        print("All faces have been embedded")
        embeddings = Xids_db

        # Normalize vectors for the later compute of l2
        in_encoder = Normalizer(norm='l2')  # Tengo que investigar que es la normalización l2
        Xids_db = in_encoder.transform(Xids_db)

        print("The embeddings have been normalized")

        # Different labels to integers for svc
        multiclasser = LabelEncoder()
        multiclasser.fit(Yids_db)  # Learning the classes
        Ytrain_labeled = multiclasser.transform(Yids_db)  # Labeling in integers

        print("The labels have been encoded")
        print("Y train =", Yids_db)

        # Training the model
        # SUPORT VECTOR MACHINE, optional classifier

        # Creating the vector through data train
        svc = SVC(kernel='linear', probability=True)
        svc.fit(Xids_db, Ytrain_labeled)

        print("The classifier has been initialized")

        return svc, Xids_db, Yids_db, embeddings

    def l2_distance_evaluation(self, yhatl2, embeddings, Ytrain, mode = 2, id=None, vp=None, vn=None, fp=None, fn=None):
        eval = [0, 100, "No Face Found"]
        in_encoder = Normalizer(norm='l2')

        # Getting the embeddings of the Xids_db in the data base
        for index, e in enumerate(embeddings):
            e = [e, ]
            e = in_encoder.transform(e)


            yhatl2 = yhatl2.reshape(1,-1)
            yhatl2 = in_encoder.transform(yhatl2)
            # Computing euclidean distance between the detected face and the current embedd
            eval[0] = numpy.linalg.norm(in_encoder.transform(yhatl2)-in_encoder.transform(e))
            print( "Distancia a normalizada", Ytrain[index], " es de ", eval[0])

            if mode == 0:
                if eval[0] <= 1.248:
                    if (id == Ytrain[index]) & (id is not None):
                        vp += 1
                        print(Ytrain[index] + " es un vp")
                    if (id != Ytrain[index]) & (id is not None):
                        fp += 1
                        print(Ytrain[index] + " es un fp")
                if eval[0] > 1.248:
                    if (id == Ytrain[index]) & (id is not None):
                        fn += 1
                        print(Ytrain[index] + " es un fn")
                    if (id != Ytrain[index]) & (id is not None):
                        vn += 1
                        print(Ytrain[index] + " es un vn")

            if eval[0] <= eval[1]:
                eval[1] = eval[0]
                eval[2] = Ytrain[index]
        if mode == 1:
            for index, identity in enumerate(Ytrain):
                if Ytrain[index] != id:
                    if eval[2] != Ytrain[index]:
                        vn += 1
                    if eval[2] == Ytrain[index]:
                        fp += 1
                if Ytrain[index] == id:
                    if Ytrain[index] == eval[2]:
                        vp += 1
                    if Ytrain[index] != eval[2]:
                        fn += 1

        if eval[1] > 1.242:
            eval[2] = "Unknown"
        if id is None:
            return eval
        else:
            return eval, vp, vn, fp, fn

    def show_fram(self, frame):
        frame = np.uint8(frame)
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) == ord('q'):
            return False
        else:
            return True
    def new_image(self,frame, k, count):
        if count == 0:
            print(" With enough light, approximate your face to the camera")
            print(" Press 'a' to take a photo ")
            count = 1
        if k == ord('a'):
            cv2.destroyAllWindows()
            self.show_fram(frame)
            print(" Press 'a' again to confirm")
            print(" Ensure that your face is without problems")
            k = cv2.waitKey(0)
            if k == ord('a'):
                flag = self.new_identity(frame, k)
                return flag, count
        else:
            flag = True
            return flag, count


    def new_identity(self, frame, k):

        print("Write your name with '_' instead of spaces")
        id = input()
        flag = True
        while flag:
            print("Your name is ", id, "is that correct?")
            print("[y]")
            print("[n]")
            k = cv2.waitKey(0)
            if k == ord('n'):
                self.new_identity(frame, k)
            if k == ord('y'):
                flag = self.store_new_identity(frame, id, flag)
                return flag


    def store_new_identity(self, frame, id, flag):

        frame = np.array(frame)
        directory = 'Real_Data/Train/' + id
        if not os.path.exists(directory):
            os.makedirs(directory)
            directory = directory + '/' + id + '.jpg'
            cv2.imwrite(directory, frame)
            if not os.path.exists(directory):
                print("Something went wrong, please try again")
                flag = True
                return flag
            else:
                print("You have been sucessfully added to the system")
                flag = False
                return flag

        else:
            print("Your id is already on the System")
            flag = False
            return flag

    def menu(self, streaming):
        print("Write '0' to add a new identity into the database")
        print("Write '1' to validate the system")
        print("Write '2' to start facial recognition though client")
        selection = input()
        if selection == '0':
            count = 0
            self.take_new_image(streaming, count)
            self.menu(streaming)
        if selection == '1':
            self.validate_menu(streaming)
        if selection == '2':
            self.face_recognition(streaming)
            self.menu(streaming)
        else:
            self.menu(streaming)
    def take_new_image(self,streaming, count):

        server, connection = streaming.server_init()
        flag = True
        try:
            while flag:
                frame = streaming.video_streaming(connection)
                if frame == False:
                    break
                k = cv2.waitKey(5)
                flag, count = self.new_image(frame, k, count)
                if self.show_fram(frame) == False:
                    break
        finally:
            streaming.ending_streaming(connection, server)

    def face_recognition(self, streaming):

        face_detector = MTCNN()

        # Get the svc and embeddings with the dataset on Real_Data/
        svc, Xtrain, Ytrain, embeddings = self.train_classifier_with('Real_Data/')

        embedding_model = load_model('facenet_keras.h5')
        in_encoder = Normalizer(norm='l2')
        font = cv2.FONT_HERSHEY_SIMPLEX
        server, connection = streaming.server_init()
        try:

            while True:

                frame = streaming.video_streaming(connection)
                if frame == False:
                    break
                face = self.extract_face_webacm(frame, face_detector)

                # Checking if there has been detected a face by the size of the image/

                if face.shape == (160, 160, 3):  # Checking array 160x160

                    # Lets compute the embeddings

                    yhat = self.embed_face(embedding_model, face)

                    # Lets classifie the embedd

                    yhatl2 = self.l2_distance_evaluation(yhat, embeddings, Ytrain)
                    yhatsvc = [yhat, ]
                    yhatsvc = in_encoder.transform(yhatsvc)
                    yhatsvc = svc.predict(yhatsvc)
                    yhatsvc = str(yhatsvc)
                    frame = np.uint8(frame)
                    cv2.putText(frame, yhatsvc, (10, 30), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                    cv2.putText(frame, yhatl2[2], (100, 30), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                    print("SVC result:", yhatsvc)
                    print("L2 distance:", yhatl2[2], "with similitude of", yhatl2[1])  # algo en caso yhat null

                # Showing the frame
                if not self.show_fram(frame):
                    break
        finally:
            streaming.ending_streaming(connection, server)

        cv2.destroyAllWindows()

    def validate_menu(self, streaming):
        print("Write '0' to record a new video for the test set")
        print("Write '1' to validate the systems using the idendities on the DB and the recorded videos")
        selection = input()
        if selection == '0':
            count = 0
            self.record_video(streaming)
            self.menu(streaming)
        if selection == '1':
            self.configure_validation()

    def record_video(self, streaming):
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        print(" Please write down the name of the person who will be recorded using '_' instead of spaces")
        id = input()
        print(" The video will be stored in folder named", id, "in path /val/")
        directory = 'val/' + id
        if not os.path.exists(directory):
            os.makedirs(directory)
            name = id + '_0'
        else:
            celebrities = os.listdir(directory)
            number = len(celebrities)
            name = id + '_' + str(number)
        print(" The streaming will start, please press 'a' whenever you want to start recording")
        print(" Press 'a' again to stop the video and store it ")
        count = 0
        server, connection = streaming.server_init()
        try:
            while True:
                frame = streaming.video_streaming(connection)
                if frame == False:
                    break
                frame = np.array(frame)
                if count == 1:
                    writer.write(frame)
                k = cv2.waitKey(5)
                if (k == ord('a')) & (count == 0):
                    print("Video started to be recorded")
                    width = frame.shape[1]
                    height = frame.shape[0]
                    writer = cv2.VideoWriter(directory + '/' + name +'.avi', fourcc, 5, (width, height))
                    count = 1
                    k = 0
                if (k == ord('a')) & (count == 1):
                    writer.release()
                    print("Video finished")
                    cv2.destroyAllWindows()
                    break

                if not self.show_fram(frame):
                    cv2.destroyAllWindows()
                    break

        finally:
            cv2.destroyAllWindows()
            print("Video succesfully stored")
            streaming.ending_streaming(connection, server)

    def configure_validation(self):

        directory = 'val/'
        streaming, face_detector, svc, Xtrain, Ytrain, embeddings, embedding_model, in_encoder, font = self.initialize_face_recognition_objects()
        print(" The following identities, have been found ")
        print(Ytrain)
        identities_with_videos = os.listdir(directory)
        print(" There exist videos for the following identities")
        print (identities_with_videos)
        unknown = []
        # Checking the unkown identitys shown in the videos
        for v in identities_with_videos:
            check = 0
            for i in Ytrain:
                if v == i:
                    check = 1
            if check == 0:
                unknown.append(v)

        print(" The following idendities are recorded on the videos without its identitis being recorded on the DB of identities")
        print( " These should be identified as 'Unknown' ")
        print( unknown)

        print( " Now, the recorded videos in path val/ will be processed")
        print( " After it, a report with the results will be shown ")

        # Creating all the variables needed for the storage of the results

        svc_video_results = []
        l2_video_results = []
        number_of_predictions = 0
        vp = 0
        vn = 0
        fp = 0
        fn = 0
        embedd_time = []
        detection_time = []
        clasification_time = []
        iteration_time = []
        for id in identities_with_videos:
            directory_folder_video = directory + id
            videos_of_id_X = os.listdir(directory_folder_video)
            print(videos_of_id_X)
            for index, v in enumerate(videos_of_id_X):
                print(len(videos_of_id_X))
                print(index)

                directory_video = directory_folder_video + '/' + str(id) + '_' + str(index) + '.avi'
                video = cv2.VideoCapture(directory_video)
                print("Extracting Video " + str(index) + ' of ' + str(id))
                if video.isOpened():
                    flag = True
                else:
                    flag = False

                while flag:
                    ret, frame = video.read()
                    if not ret:
                        print("empty")
                        flag = False
                        continue
                    start_iteration = time.time()
                    start_detection = time.time()
                    face = self.extract_face_webacm(frame, face_detector)
                    end_detection = time.time()

                    if face.shape == (160, 160, 3):  # Checking array 160x160

                        # Lets compute the embeddings
                        start_embedd= time.time()
                        yhat = self.embed_face(embedding_model, face)
                        end_embedd = time.time()
                        start_clasification = time.time()
                        yhatl2, vp, vn, fp, fn = self.l2_distance_evaluation(yhat, embeddings, Ytrain, 1, str(id), vp, vn, fp, fn )
                        end_clasification = time.time()
                        yhatsvc = [yhat, ]

                        yhatsvc = in_encoder.transform(yhatsvc)

                        # Lets classifie the embedds

                        yhatsvc = svc.predict(yhatsvc)
                        yhatsvc = str(yhatsvc)
                        frame = np.uint8(frame)
                        cv2.putText(frame, yhatsvc, (10, 30), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                        cv2.putText(frame, yhatl2[2], (100, 30), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                        print("SVC result:", yhatsvc)
                        svc_video_results.append(yhatsvc)
                        print("L2 distance:", yhatl2[2], "with similitude of ", yhatl2[1])  # algo en caso yhat null
                        l2_video_results.append(yhatl2[2])

                        end_iteration = time.time()
                        detection_time.append(end_detection-start_detection)
                        embedd_time.append(end_embedd-start_embedd)
                        clasification_time.append(end_clasification-start_clasification)
                        iteration_time.append(end_iteration-start_iteration)
                        number_of_predictions += 1

                    # Showing the frame
                    if not self.show_fram(frame):
                        continue

                print("here")
                print("Video " + str(index) + ' of ' + str(id) + ' finished')

        cv2.destroyAllWindows()
        if number_of_predictions == 0:
            print("No predictions")
        else:
            print(" Results computed taking in consideration the described metrics are:")
            print(" Number of True Positive: " + str(vp))
            print(" Number of True Negatives: " + str(vn))
            print(" Number of False Positives: " + str(fp))
            print(" Number of False Negatives: " + str(fn))

            print(" The system shows an accuracy of: " + str((vp+vn)*100/(vp+vn+fn+fp)) + " %")
            print(" The system shows an precision of: " + str((vp) * 100 / (vp + fp)) + " %")
            print(" The system shows an sensibility of: " + str((vp) * 100 / (vp + fn)) + " %")
            print(" The system shows an specificity of: " + str((vn) * 100 / (vn + fp)) + " %")

            print(" The average time per frame is: " + str(sum(iteration_time)/len(iteration_time)) + "s")
            print(" Overall frame per second is: " + str(1/(sum(iteration_time)/len(iteration_time)))+ "s")
            print(" The average time spend in the face detector is: " + str(sum(detection_time) / len(detection_time))+ "s")
            print(" The average time spend in the embedding of the faces is: " + str(sum(embedd_time) / len(embedd_time))+ "s")
            print(" The average time spend in the clasificator is: " + str(sum(clasification_time) / len(clasification_time))+ "s")

    def initialize_face_recognition_objects(self, mode = 0):
        streaming = self.streaming
        face_detector = MTCNN()

        # Get the svc and embeddings with the dataset on Real_Data/
        svc, Xids_db, Yids_db, embeddings = self.train_classifier_with('Real_Data/')
        embedding_model = load_model('facenet_keras.h5')
        in_encoder = Normalizer(norm='l2')
        font = cv2.FONT_HERSHEY_SIMPLEX
        if mode == 1:
            server, connection = streaming.server_init()
            return streaming, server, connection, face_detector, svc, Xids_db, Yids_db, embeddings, embedding_model, in_encoder, font
        else:
            return streaming, face_detector, svc, Xids_db, Yids_db, embeddings, embedding_model, in_encoder, font