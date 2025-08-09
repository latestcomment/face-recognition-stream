import cv2

from threading import Thread

class Stream():
    def __init__(self, stream_address):
        self.stream_address = stream_address

        # open video capture
        self.vcap      = cv2.VideoCapture(self.stream_address, cv2.CAP_AVFOUNDATION)
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5)) # hardware fps
        self.width = int(self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # reading a single frame from vcap stream for initializing 
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)

        self.stopped = True

        self.t = Thread(target=self.update, args=())
        self.t.daemon = True

    def start(self):
        self.stopped = False
        self.t.start()

    def update(self):
        while True:
            if self.stopped is True :
                break

            self.grabbed , self.frame = self.vcap.read()
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break
        self.vcap.release()

    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True