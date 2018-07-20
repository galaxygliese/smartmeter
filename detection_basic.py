#-*- coding:utf-8 -*-

from sklearn.neighbors import LocalOutlierFactor
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import timeit
import json
import cv2

#Find the biggest shape
def max_shape(array, _type ):
    point = np.array(array)
    if _type in ['area', 'rect']:
       areas = []
       for i,rect in enumerate(point):
            areas.append(rect[2]*rect[3])
       return point[areas.index(max(areas))]
    elif _type in ['r', 'radius']:
         #input array included radius in third index
         max_radius = np.max([r[2] for r in point])
         return [r for r in point if r[2]==max_radius][0]



#Distance function between points
def dist(self, x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)



#Detect Gauge
class GaugeDetection:
      def __init__(self):
          self.cascade_path = "./only_pressure_gauge14_80.xml"
          self.cascade = cv2.CascadeClassifier(self.cascade_path)
          self.gray = None
          self.LofLength = 30
          #Layers
          self.cascades_output = None
          self.circles_output = None
          self.center_output = None
          self.line_output = None
          #Check outputs
          self.cascades_run = False
          self.circles_run = False
          self.center_run = False
          self.line_run = False
          #Parameters
          ###CASCADE
          self._x, self._y = None, None
          self._w, self._h = None, None
          ###CIRCLES
          self.Radius = deque(maxlen=self.LofLength)
          self.R = None
          ###CENTER
          self.Centers = deque(maxlen=self.LofLength)
          self.clf = LocalOutlierFactor(n_neighbors=2)
          self.X, self.Y = None, None 
          ###LINE
          self.k = None
          ###SHOW
          self.d = 5      

      def run(self, frame):
          self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          try:
             self.cascades()
             self.circles()
             self.center()
             self.line()
          except Exception as e:
             print(e)

      def cascades(self):
          self.point = self.cascade.detectMultiScale(self.gray, 1.1, 3)
          if len(self.point) > 0:
             rect = max_shape(self.point, 'area')
             #max square  _x, _y --> coordinate of the detected rectangle
             _x, _y = int(rect[0]), int(rect[1])
             _w, _h = int(rect[2]), int(rect[3])
             
             self._x, self._y = _x, _y
             self._w, self._h = _w, _h
             self.cascades_output = (_x, _y, _w, _h)
             self.cascades_run = True
          else:
             self.cascades_run = False

      def circles(self):
          if self.cascades_run:
             #in_cascade = self.gray[self._y:self._y+self._h, self._x:self._x+self._w]
             circles = cv2.HoughCircles(self.gray, cv2.HOUGH_GRADIENT, 1.2, 100) 
             if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                x, y ,r  = max_shape(circles, 'radius')
                if (self._x< x < self._x+self._w) and (self._y < y < self._y+self._h):
                   self.Radius.append(r)
                   self.circles_output = (x, y, r)
                   self.R = int(r)
                   self.circles_run = True
             else:
                self.circles_run = False
          else:
             pass 
                   
      def center(self):
          if self.circles_run:
             x, y, r = self.circles_output
             self.Centers.append([x,y])
             centers = np.array(self.Centers)
             if len(centers) > self.LofLength-1:
                predict = self.clf.fit_predict(centers)
                true_center = centers[predict==1]
                X, Y = np.mean(true_center[:,0]), np.mean(true_center[:,1])
             else:
                X, Y = np.mean(centers[:,0]), np.mean(centers[:,1])
             self.X, self.Y = int(X), int(Y)
             
      def line(self):
          self.line_run = True
 
      def show(self):
         if (self.X, self.Y) != (None, None):
             cv2.circle(self.gray, (self.X, self.Y), self.R, (0, 255, 0), 4)
             cv2.rectangle(self.gray, (self.X - self.d, self.Y - self.d), (self.X +
 self.d, self.Y + self.d), (0, 255, 0), -1)
         cv2.imshow('frame', self.gray)

      def save(self, delta_t, freq):#-->json bolgoh
          if delta_t % self.freq < self.eps:
             with open('params.k', 'w') as f:
                  f.write(self.k)


if __name__ == '__main__':
   from billiard import Process, forking_enable
   def start():
       forking_enable(0)
       camProcess = Process(target=cam, args=(0,))
       camProcess.start()
   def cam(cam_id):
       #initial time
       time = 15
       start = timeit.default_timer()
       cap = cv2.VideoCapture(cam_id)
       ret, frame = cap.read()
       init_detect = GaugeDetection()
       detect = GaugeDetection()

       #center coordinate
       X, Y, R = None, None, None
       while(cap.isOpened()):
           ret, frame = cap.read()
           if ret:
              stop = timeit.default_timer()
              if stop-start <= time :
                init_detect.run(frame)
                init_detect.show()
                #get final parameters
                X, Y = init_detect.X, init_detect.Y
                R = max(init_detect.Radius) if any(init_detect.Radius) else 0
                R = int(R) 
              else:
                detect.run(frame[Y-R:Y+R, X-R:X+R])
                detect.show()
           k = cv2.waitKey(10)
           if k == 27:
              break
           if k == ord('q'):
              break
       cap.release()
       cv2.destroyAllWindows()

   start()
   cam(0)
          


