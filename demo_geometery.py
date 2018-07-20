#-*- coding:utf-8 -*-

from sklearn.neighbors import LocalOutlierFactor
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import timeit
import cv2


class detectGauge:

      def __init__(self):
          self.cascade_path = "./only_pressure_gauge14_80.xml"
          self.gray = None
          #CENTER 
          self.LENGTH = 30 #LOF array length
          self.d = 5 #center rectangle's w and h       
          self.X, self.Y = None, None
          self.Radius = []
          self.Center = deque(maxlen=self.LENGTH)
          self.clf = LocalOutlierFactor(n_neighbors=2)
          #LINE
          self.LinePara = 10 #LOF array length
          self.tan_theta = deque(maxlen=self.LinePara)
          self.LineLof = LocalOutlierFactor(n_neighbors=2)
          self.k = None
          self.MinusX = False
          self.MinusY = False
          self.points = deque(maxlen=self.LinePara)
          self.kernel = np.ones((2,2),np.uint8)
          #SAVE DATA
          self.freq = 10 #write frequency [sec]
          self.eps = 2 #delta_t error

      #Find the biggest shape
      def max_shape(self, array, _type ):
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

      #Find quadrant
      def quadrant(self, x, y):
          if x<self.X:
             return 1 if y < self.Y else 3
          else:
             return 2 if y < self.Y else 4

      def line_weighting(self, x1, y1, x2, y2):
          d1 =  self.dist(x1, y1, self.X, self.Y)
          d2 =  self.dist(x2, y2, self.X, self.Y)
          d = d1 + d2
          return (d1/d, d2/d)
          


      #Detection of the line
      def line(self, x, y, r):
          kernel = self.kernel
          thresh = 175
          maxValue = 255
          img = self.gray
          #th, dst2 = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY_INV);
 
          h, w = img.shape[:2]
          mask = np.zeros((h+2, w+2), np.uint8)
          th, dst2 = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY_INV)

          minLineLength = 10
          maxLineGap = 0
          lines = cv2.HoughLinesP(image=dst2, rho=3, theta=np.pi / 180, threshold=100,minLineLength=minLineLength, maxLineGap=0)

          kernel = np.ones((2,2),np.uint8)
          white = np.full(img.shape, 255)
          thresh = 175
          maxValue = 255
 

          final_line_list = []

          diff1LowerBound = 0.15     
          diff1UpperBound = 0.25
          diff2LowerBound = 0.5   
          diff2UpperBound = 1.0
          for i in range(0, len(lines)):
              for x1, y1, x2, y2 in lines[i]:
                  diff1 = self.dist(x, y, x1, y1)  # x, y is center of circle
                  diff2 = self.dist(x, y, x2, y2)  # x, y is center of circle
                  #set diff1 to be the smaller (closest to the center) of the two), makes the math easier
                  if (diff1 > diff2):
                    temp = diff1
                    diff1 = diff2
                    diff2 = temp
                  # check if line is within an acceptable range
                  if (((diff1<diff1UpperBound*r) and (diff1>diff1LowerBound*r) and (diff2<diff2UpperBound*r)) and (diff2>diff2LowerBound*r)):
                    line_length = self.dist(x1, y1, x2, y2)
                  # add to final list
                    final_line_list.append([x1, y1, x2, y2])
          if final_line_list != []:
             x1 = final_line_list[0][0]
             y1 = final_line_list[0][1]
             x2 = final_line_list[0][2]
             y2 = final_line_list[0][3]
   
             self.points.append([x1, y1, x2, y2])

             #tan(theta)
             if x2 != x1: #tan(pi/2) = inf
                k = (y2-y1)/(x2-x1)
                self.tan_theta.append(k)
                K_params = np.array(self.tan_theta)
                Points = np.array(self.points)
             #LOF
                if len(self.tan_theta) > self.LinePara/10:
                   predict = self.LineLof.fit_predict([[i] for i in K_params]) 
                   self.k = np.mean(K_params[predict==1])

                   if np.min([x1,x2]) < self.X and np.min([y1,y2]) < self.Y:
                      m, n = self.line_weighting(x1, y1, x2, y2)
                      conditionX = (m*Points[:,0]+n*Points[:,2]) < self.X #mean of the two points is smaller than the center
                      conditionY = (m*Points[:,1]+n*Points[:,3]) < self.Y #mean of the two points is smaller than the center                  
                
                      #Compute probably of the position     
                      self.MinusX = True if np.sum(conditionX) > 0.5 else False
                      self.MinusY = True if np.sum(conditionY) > 0.5 else False
                   else:
                      self.MinusX = False
                      self.MinusY = False         
          
                   #self.MinusX = True if np.min([x1, x2]) < self.X and np.min([y1, y2]) < self.Y else False
                   #self.MinusY = True if np.min([x1, x2]) < self.X and np.min([y1, y2]) < self.Y else False
                      
             else:
                self.k = 'inf'
       
      #The main process  
      def run(self, frame):
          #gray
          self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)         
          #cascade
          self.cascade = cv2.CascadeClassifier(self.cascade_path)
          #detection
          self.point = self.cascade.detectMultiScale(self.gray, 1.1, 3)

          if len(self.point) > 0:
           rect = self.max_shape(self.point, 'area')

           #max square  _x, _y --> coordinate of the detected rectangle
           _x, _y = int(rect[0]), int(rect[1])
           _w, _h = int(rect[2]), int(rect[3])          
            
           #detect circle x, y, X, Y --> coordinate of the biggest circle
           circles = cv2.HoughCircles(self.gray, cv2.HOUGH_GRADIENT, 1.2, 100)
           if circles is not None:
              circles = np.round(circles[0, :]).astype("int")
              x, y ,r  = self.max_shape(circles, 'radius')
              self.Center.append([x,y])
              self.Radius.append(r)
              center = np.array(self.Center)
           #find LOF and find Center
              if len(center) > self.LENGTH-1:                 
                 predict = self.clf.fit_predict(center) 
                 true_center = center[predict==1]   
                 X, Y = np.mean(true_center[:,0]), np.mean(true_center[:,1])
              else:
                 X, Y = np.mean(center[:,0]), np.mean(center[:,1])
           #check circles in cascade detection
              if (_x < X < _x+_w) and (_y < Y < _y+_h):                             
                  self.X, self.Y, self.r = int(X), int(Y), int(r)
                  self.line(self.X, self.Y, self.r)
      #Save output
      def save(self, delta_t, freq):
          if delta_t % self.freq < self.eps:
             with open('datas.txt', 'w') as f:
                  f.write(self.k)
 
      #Show on a display
      def show(self):
          if (self.X, self.Y) != (None, None): 
             cv2.circle(self.gray, (self.X, self.Y), self.r, (0, 255, 0), 4)
             cv2.rectangle(self.gray, (self.X - self.d, self.Y - self.d), (self.X + self.d, self.Y + self.d), (0, 255, 0), -1)
             #Line
             try:
                if self.k != None:
                   if self.k != 'inf':
                         x1 = np.int(self.X + self.r/np.sqrt(1+self.k**2))
                         x2 = np.int(self.X - self.r/np.sqrt(1+self.k**2))
                         y1 = np.int(self.Y + self.k*self.r/np.sqrt(1+self.k**2))
                         y2 = np.int(self.Y - self.k*self.r/np.sqrt(1+self.k**2))
                      
                   else:
                      x2 = self.X
                      y2 = self.r
                   cv2.line(self.gray, (x1, y1), (x2, y2), (0, 255, 0), 2)
             except Exception :
                 pass
          cv2.imshow('frame', self.gray)
 
              

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
       init_detect = detectGauge()
       detect = detectGauge() 

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
              else:
                detect.run(frame[Y-R:Y+R, X-R:X+R])
                detect.show()
                if timeit.default_timer() - start >= 2000:
                   detect.R=deque(maxlen=10)
           k = cv2.waitKey(10)
           if k == 27:
              break
           if k == ord('q'):
              break
       cap.release()
       cv2.destroyAllWindows()

   start()
   cam(0)

