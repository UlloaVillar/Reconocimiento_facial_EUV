import io
import socket
import struct
import time
import picamera

PORT = 0
client_socket = socket.socket()
client_socket.connect(("IP"),PORT)

# Creating a file out of the connection on write mode 'wb'

connection = client_socket.makefile('wb')
try:
     camera = picamera.PiCamera()
     camera.vflip = True
     camera.resolution = (500,480) #MTCNN requires at least 160x160
     camera.start_preview()
     #Leet the camera start itself
     time.sleep(3)

     # Streaming of bytes
     stream = io.BytesIO()
     for foo in camera.capture(stream, 'jpeg'): # foo means anything
         #El size viene dado por el formato
        connection.write(struct.pack('<L', stream.tell())) #L means from right to left #struct to interpet bytes as packed bynary
        connection.flush() # to send the data not sent
        stream.seek(0) #Go to the start of the stream
        connection.write(stream.read())
        stream.seek(0)
        stream.truncate() #Reset the stream to the current position () Sending frame by frame

finally:
    connection.close()
