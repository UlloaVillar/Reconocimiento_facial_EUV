import io
import socket
import struct
from PIL import Image


class Streaming:


    def server_init(self):
        server = socket.socket()
        server.bind(('', 8000))
        server.listen(0)
        print("Waiting for connection")

        # Accept connection and create a read object out of it
        connection = server.accept()[0].makefile('rb')
        print("Connection accepted")
        return(server, connection)

    def video_streaming(self, connection):

        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            return False
        # Constructing the stream to read the data image
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        # Rewind the stream, open it as an image with PIL and do some
        # processing on it
        image_stream.seek(0)
        image = Image.open(image_stream)
        return image


    def ending_streaming(self,server, connection):

            connection.close()
            server.close()