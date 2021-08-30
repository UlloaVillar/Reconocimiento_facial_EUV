
import tensorflow as tf
import keras



tf_config = tf.ConfigProto( allow_soft_placement=True )
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
keras.backend.set_session(sess)


from mtcnn.mtcnn import MTCNN

from FaceRecognition import FaceRecognition
from server import Streaming


face_processing = FaceRecognition()
streaming = Streaming()
face_detector = MTCNN()



election = face_processing.menu(streaming)

