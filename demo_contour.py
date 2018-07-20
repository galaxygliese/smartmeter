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
def dist(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

#Point in Circle
def in_circle(x, y, cx, cy, r):
    return True if dist(x, y, cx, cy) < r else False

#Find neatest point
def nearest(points, cx, cy):
    dists = []
    for x, y in points:
        dists.append(dist(x ,y, cx, cy))
    index = dists.index(min(dists))
    return index

#Farest point 
def farest(rect, cx, cy):
    x, y, w, h = rect
    dist1 = dist(x, y, cx, cy)
    dist2 = dist(x+w, y, cx, cy)
    dist3 = dist(x, y+h, cx, cy)
    dist4 = dist(x+w, y+h, cx, cy)
     
    dist12 = int(dist1) if dist1>dist2 else int(dist2)
    dist34 = int(dist3) if dist3>dist4 else int(dist4)
    Fdist = dist12 if dist12 > dist34 else dist34
  
    if Fdist == int(dist1):
       return (x, y , w, h)
    if Fdist == int(dist2):
       return (x+w, y, w, h)
    if Fdist == int(dist3):
       return (x, y+h, w, h)
    else:
       return (x+w, y+h, w, h)

#Farest point between 2    
def farest2point(point1, point2, cx, cy):
    dist1 = dist(point1[0], point1[1], cx, cy) 
    dist2 = dist(point2[0], point2[1], cx, cy)
    return 0 if dist1 > dist2 else 1

#FInd blackest rectangle 
def blackest(img, rect1, rect2):
    mean = np.mean(img)
    rec1 = img[rect1[1]:rect1[1]+rect1[3], rect1[0]:rect1[0]+rect1[2]]    
    rec2 = img[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]]    

    black1 = np.sum(rec1<mean) / (rec1.shape[0]*rec1.shape[1])
    black2 = np.sum(rec2<mean) / (rec2.shape[0]*rec2.shape[1])
    return 0 if black1 > black2 else 1



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
          self.numbers_run = False
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
          self.a = 1.8 #=> radius zoom parameter
          #self.LinePara = 10 #=> ...
          #self.minLineLength = 50
          #self.diagnal_range = 0.6 
          self.MinPerimeter = 100
          self.Lx, self.Ly = None, None
          self.Lw, self.Lh = None, None
          self.x1, self.y1 = None, None
          self.x2, self.y2 = None, None
          self.Px, self.Py = None, None
          self.PX, self.PY = None, None
          self.point_len = 20
          #self.MinLen = 10 #=> number of the frame
          #self.box = deque(maxlen=self.LofLength)
          self.points = deque(maxlen=self.point_len)
          self.tan_theta = deque(maxlen=self.LofLength)
          self.LineLof = LocalOutlierFactor(n_neighbors=2)
          self.k = None
          ###NUMBERS
          self.contours = None
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
             self.circles_run = False
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
        if self.circles_run:
           center_X, center_Y = self.X, self.Y

           #retval, th = cv2.threshold(self.gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
           bw = cv2.adaptiveThreshold(self.gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)


           im2, self.contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
           detect_count = 0
           coordinates = []
           rects = []
           if len(self.contours) > 0:
              for i in range(0, len(self.contours)):
                  area = cv2.contourArea(self.contours[i])
                  if area < 1e2 or 1e5 < area:
                     continue
 
                  if len(self.contours[i]) > 0:
                     rect = self.contours[i]
                     x, y, w, h = cv2.boundingRect(rect)
                     #if w < self.R and h < self.R:
                     if in_circle(x, y, self.X, self.Y, self.a*self.R) and in_circle(x+w, y+h, self.X, self.Y, self.a*self.R) and (w**2+h**2<(self.a*self.R)**2) and (x < self.X < x+w) and (y<self.Y<y+h) and w+h > self.MinPerimeter:   
                        #if dist(x, y, self.X, self.Y) < dist(x+w, y+h, self.X, self.Y):
                        coordinates.append([x, y])
                        #else:
                        #coordinates.append([x+w, y+h])
                        rects.append([x, y, w, h])
                     detect_count = detect_count + 1
              if coordinates != []:
                 self.Lx, self.Ly, self.Lw, self.Lh = max_shape(rects, 'rect')
                 
    
                 ##left side
                 if self.X-self.Lx < self.Lw/2:
                    W = self.Lx+self.Lw-self.X
                    im = self.gray[self.Ly:self.Ly+self.Lh, self.X:self.Lx+self.Lw]
                    #blackest=0 -> upper 
                    #if blackest(im, [self.X, self.Ly, W, self.Y-self.Ly], [self.X, self.Y, W, self.Ly+self.Lh-self.Y ]) == 0:
                    if self.Y-self.Ly>self.Lh/2:
                       self.points.append([self.Lx+self.Lw, self.Ly])
                    else:
                       self.points.append([self.Lx+self.Lw, self.Ly+self.Lh])
                 ##right side
                 else:
                    W = self.X-self.Lx
                    im = self.gray[self.Ly:self.Ly+self.Lh, self.Lx:self.X]
                    #blackest=0 -> upper 
                    #if blackest(im, [self.Lx, self.Ly, W, self.Y-self.Ly], [self.Lx, self.Y, W, self.Ly+self.Lh-self.Y]) == 0:
                    if self.Y-self.Ly>self.Lh/2:
                       self.points.append([self.Lx, self.Ly])
                    else:
                       self.points.append([self.Lx, self.Ly+self.Lh])
 
                 self.Px = self.points[-1][0]        
                 self.Py = self.points[-1][1]        
            
                 Points = np.array(self.points)
                 k = (self.Py-self.Y)/(self.Px-self.X)                
                 self.tan_theta.append(k)
                 K_params = np.array(self.tan_theta)
                 if len(self.tan_theta) > 1:   #self.LinePara/10: 
                    predict = self.LineLof.fit_predict([[i] for i in K_params]) 
                    self.k = np.mean(K_params[predict==1])

                    predict = self.LineLof.fit_predict(Points)
                    true_Points = Points[predict==1]
                 
                    self.PX = int(np.mean(true_Points[:,0]))
                    self.PY = int(np.mean(true_Points[:,1]))
                    #self.k = k
                    #self.Px = np.int(self.X + self.R/np.sqrt(1+self.k**2))
                    #self.Py = np.int(self.Y + self.k*self.R/np.sqrt(1+self.k**2))
                      
              else :
                    self.line_run = False
           else: 
             self.line_run = False
        else:
            self.line_run = False
            pass
 
      def numbers(self):
          if len(self.contours ) > 0:
             number_coord = []
             for i in range(0, len(self.contours)):
                 area = cv2.contourArea(self.contours[i])
                 if area < 1e2 or 1e5 < area:
                    continue
                 if len(self.contours[i]) > 0:
                    rect = self.contours[i]
                    x, y, w, h = cv2.boundingRect(rect)
            
          else:
             self.numbers_run = False

      def show(self):
         if (self.X, self.Y) != (None, None):
             if self.R != None:
                cv2.circle(self.gray, (self.X, self.Y), self.R, (0, 255, 0), 4)
                cv2.rectangle(self.gray, (self.X - self.d, self.Y - self.d), (self.X + self.d, self.Y + self.d), (0, 255, 0), -1)
             if (self.Lx, self.Ly, self.Lw, self.Lh) != (None, None, None, None):
                cv2.rectangle(self.gray, (self.Lx, self.Ly), (self.Lx + self.Lw, self.Ly + self.Lh), (0, 255, 0), 2)
             if (self.PX, self.PY) != (None, None):  
                cv2.line(self.gray, (self.PX, self.PY), (self.X, self.Y), (0, 255, 0), 2)
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
       time = 10
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
          


