import socket
import pickle
import cv2
import numpy as np
import os
import time
from . import ProductDetection
from . import Tracking


class ServerSocket:

    def __init__(self,IP,port):
        self.IP=IP
        self.port=port
        self.serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.buffer_size = 4096
        self.tracker=Tracking.EuclideanDistTracker()

    def ConnectSocket(self):
        self.serv.bind((self.IP,self.port))
        self.serv.listen(5)

    def SendRecieveSocket(self):
        while True:
            print("Waiting to connect")
            conn, addr = self.serv.accept()
            print("Connection on " + str(addr))
            data = b''
            while True:
                received = conn.recv(self.buffer_size)
                data += received
                if len(received) < self.buffer_size:
                    break
            file = open("image.jpg", "w+b")
            file.write(data)
            file.close()
            detection = ProductDetection.ProductDetection();
            response = bytes(detection.videoDetector("image.jpg", self.tracker), 'utf-8')
            # processing
            conn.send(response)
            conn.close()




