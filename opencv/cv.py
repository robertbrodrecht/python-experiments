import cv2
import math
import numpy
import os
import PIL
from PIL import Image, ImageDraw
import time

import pprint



class Object2D:
	
	def rotate(point = (0, 0), degrees = 0, origin = (0, 0)):
		"""
		Rotate a point by a given angle around a given origin.
		
		Original by Mark Dickinson on Stack Overflow http://bit.ly/2ni9EAt
		"""
		
		if degrees == 0:
			return point
		
		print('POINT BEFORE', point, isinstance(point[0], int), degrees, origin)
		
		output = []
		
		rads = math.radians(degrees)
		
		ox, oy = origin
		
		if isinstance(point[0], int):
			single = True
			points = (point,)
			
		else:
			single = False
			points = point
		
		print('POINT AFTER',points, type(points))
		
		
		# TypeError: 'numpy.int64' object is not iterable
		for px, py in points:
			qx = ox + math.cos(rads) * (px - ox) - math.sin(rads) * (py - oy)
			qy = oy + math.sin(rads) * (px - ox) + math.cos(rads) * (py - oy)
			
			output.append((round(qx), round(qy)))
		
		if single:
			return output[0]
			
		else:
			return output
	
	
	def boxPoints(box = (0, 0, 0, 0)):
		"""
		Convert a box to a crop box.
		
		A box is a square formatted as (top, left, width, height). A crop
		box is the box as (top, left, bottom, right).
		"""
		
		top, left, width, height = box
		
		bottom = top+height
		right = left+width
		
		return (top, left, bottom, right)
	
	
	def boxCenter(box = (0, 0, 0, 0)):
		"""
		Get a box's center point.
		
		A box is a square formatted as (top, left, width, height). A center
		point is returned as (x, y)
		"""
		
		top, left, width, height = box
		
		center_x = top+int(height/2)
		center_y = left+int(width/2)
		
		return (center_x, center_y)
	
	
	def boxPolygon(box = (0, 0, 0, 0)):
		"""
		Convert a box to a polygon.
		
		A box is a square formatted as (top, left, width, height). A polygon
		is the box as four distinct points.
		"""
		
		top, left, bottom, right = Object2D.boxPoints(box)
		
		top_left = (top, left)
		top_right = (top, right)
		bottom_right = (bottom, right)
		bottom_left = (bottom, left)
		
		return (top_left, top_right, bottom_right, bottom_left)




class CVFace:
	# colors
	
	# /usr/local/Cellar/opencv/3.4.0_1/share/OpenCV/haarcascades/
	# + haarcascade_frontalface_alt2.xml 
	# + haarcascade_eye.xml
	
	classifiers = {
		'eyes': {
			'file': 'haarcascade_eye.xml',
			'settings': {
				'scaleFactor': 1.05,
				'minNeighbors': 6
			},
			'classifier': False,
		},
		'face': {
			'file': 'haarcascade_frontalface_alt2.xml',
			'settings': {
				'scaleFactor': 1.05,
				'minNeighbors': 10
			},
			'classifier': False,
		}
	}
	
	@property
	def image_cv(self):
		return 	self.imageToCV()
	
	def __init__(self, image = None):
		self.classifierLoad()
		if self.imageLoad(image):
			self.facesDetect()
	
	
	def imageLoad(self, image = None):
		self.path = False
		self.image = False
		
		if not image:
			image = self.image
		
		is_pil = Image.isImageType(image)
		is_path = isinstance(image, str) and os.path.isfile(image)
		
		if is_pil:
			self.image = image
			self.center = (int(self.image.width/2), int(self.image.height/2))
			self.path = None
		
		elif is_path:
			self.image = Image.open(image)
			self.center = (int(self.image.width/2), int(self.image.height/2))
			self.path = image
		
		return self.image
	
		
	def imageToCV(self, image = None):
		if not image:
			image = self.image
		
		image_np = self.imageToNumpy(image)
		image_cv = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
		
		return image_cv
	
	
	def imageToNumpy(self, image = None):
		if not image:
			image = self.image
		
		return numpy.asarray(image)
	
	
	def imageDrawPolygon(self):
		return img
	
	
	def classifierLoad(self):
		"""Load the classifiers."""
		
		for class_type, class_data in self.classifiers.items():
			if os.path.isfile(class_data['file']):
				classifier = cv2.CascadeClassifier(class_data['file'])
				if not classifier.empty():
					self.classifiers[class_type]['classifier'] = classifier
	
	
	def facesDetect(self):
		self.faces = []
		classifier_face = self.classifiers['face']
		
		for rotation in [0, 30, -30]:
			classifier_settings = classifier_face['settings'].copy()
			rot_image = self.image.rotate(rotation, resample=PIL.Image.BILINEAR)
			
			if self.image.width > self.image.height:
				classifier_settings['minSize'] = (
					int(self.image.width*.05),
					int(self.image.width*.05)
				)
			else:
				classifier_settings['minSize'] = (
					int(self.image.height*.05),
					int(self.image.height*.05)
				)
			
			found_faces = classifier_face['classifier'].detectMultiScale(
				self.imageToCV(rot_image),
				**classifier_settings
			);
			
			for face_box in found_faces:
				face_points = Object2D.boxPoints(face_box)
				face_polygon = Object2D.boxPolygon(face_box)
				face_center = Object2D.boxCenter(face_box)
				
				face_polygon = Object2D.rotate(face_polygon, rotation, self.center)
				face_center = Object2D.rotate(face_center, rotation, self.center)
				
				
				if  not self.hasNearbyFaceCenters(face_center):
					face = {
						'rotation': rotation,
						'crop': face_points,
						'box': face_box,
						'draw': {
							'center': face_center,
							'polygon': face_polygon
						}
					}
					
					self.faces.append(face)
			
			rot_image.close()
		
		return self.faces
	
	
	def hasNearbyFaceCenters(self, center):
		if self.image.width > self.image.height:
			range_check = int(self.image.width*.015)
		else:
			range_check = int(self.image.height*.015)
		
		return False
	
	
	def faceRegions(self):
		return 
	
	
	def eyesDetect(self):
		return
	
	
	def show(self, image = None):
		if not image:
			self.image.show()
		else:
			image.show()
	
	
		
if __name__ == "__main__":
	Object2D.rotate((200, 200), 90, (500,500))
	# Mia and Erin at Disnet
	# img = CVFace('samples/IMG_2492.jpg')
	# Mia, Anna, and Joost
	img = CVFace('samples/IMG_5493.jpg')
	draw = ImageDraw.Draw(img.image)
	for face in img.faces:
		draw.polygon(face['draw']['polygon'], outline='blue')
	img.show()
	pp = pprint.PrettyPrinter(indent=4)
	pp.pprint(img.faces)
	print(len(img.faces))