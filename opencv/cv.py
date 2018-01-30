import cv2
import math
import numpy
import os
import PIL
from PIL import Image, ImageDraw
import time

import pprint



class Object2D:
	"""Do some basic 2D math stuff."""
	
	def rotate(point = (0, 0), degrees = 0, origin = (0, 0)):
		"""
		Rotate a point by a given angle around a given origin.
		
		Original by Mark Dickinson on Stack Overflow http://bit.ly/2ni9EAt
		"""
		
		if degrees == 0:
			return point
		
		output = []
		
		rads = math.radians(degrees)
		
		ox, oy = origin
		
		if isinstance(point[0], (list, tuple)):
			single = False
			points = point
			
		else:
			single = True
			points = (point,)
		
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
	"""Do some basic CV stuff."""
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
		'eyes_left': {
			'file': 'haarcascade_lefteye_2splits.xml',
			'settings': {
				'scaleFactor': 1.05,
				'minNeighbors': 6
			},
			'classifier': False,
		},
		'eyes_right': {
			'file': 'haarcascade_righteye_2splits.xml',
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
	
	normalized_size = (200, 200)
	resample = PIL.Image.BILINEAR
	
	@property
	def image_cv(self):
		"""Property-level access to the current image as a CV."""
		return 	self.imageToCV()
	
	
	def __init__(self, image = None):
		"""Loads image and classifiers, then kicks off face detection."""
		self.classifierLoad()
		if self.imageLoad(image):
			self.facesDetect()
	
	
	def classifierLoad(self):
		"""Load the classifiers."""
		
		for class_type, class_data in self.classifiers.items():
			if os.path.isfile(class_data['file']):
				classifier = cv2.CascadeClassifier(class_data['file'])
				if not classifier.empty():
					self.classifiers[class_type]['classifier'] = classifier
	
	
	def imageLoad(self, image = None):
		"""Load an image from a path or PIL object."""
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
		"""Convert a PIL image to a CV image."""
		if not image:
			image = self.image
		
		image_np = self.imageToNumpy(image)
		# Old: cv2.COLOR_BGR2RGB
		image_cv = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
		
		return image_cv
	
	
	def imageToNumpy(self, image = None):
		"""Convert a PIL image to a numpy array."""
		if not image:
			image = self.image
		
		return numpy.asarray(image)
	
	
		
	def imageDrawSquare(self, square = (0, 0, 1, 1), image=None):
		"""Draw one or more square on an image."""
		if not image:
			image = self.image
		
		if len(square) < 1: # or not any(square):
			return image
		
		if isinstance(square[0], (list, tuple, numpy.ndarray)):
			single = False
			squares = square
			
		else:
			single = True
			squares = (square,)
		
		draw = ImageDraw.Draw(image)
		
		for square in squares:
			draw.rectangle(square, outline='blue')
			
		return image
	
	
	def imageDrawPolygon(self, poly = ((0,0), (0,1), (1,1), (1, 0)), image=None):
		"""Draw one or more polygon on an image."""
		if not image:
			image = self.image
		
		if len(poly) < 1 or not any(poly):
			return image
			
		if isinstance(poly[0][0], (list, tuple, numpy.ndarray)):
			single = False
			polys = poly
			
		else:
			single = True
			polys = (poly,)
		
		draw = ImageDraw.Draw(image)
		
		for poly in polys:			
			draw.polygon(poly, outline='blue')
			
		return image
	
	
	def imageDrawPoint(self, point = (0, 0), image = None):
		"""Draw one or more points on an image."""
		if not image:
			image = self.image
		
		if len(point) < 1 or not any(point):
			return image
		
		if isinstance(point[0], (list, tuple, numpy.ndarray)):
			single = False
			points = point
			
		else:
			single = True
			points = (point,)
		
		draw = ImageDraw.Draw(image)
		
		for point in points:
			x, y = point
			
			draw.ellipse(
				(
					(x-2, y-2), 
					(x+4, y+4)
				), 
				fill='blue'
			)
			
		return image
	
	
	def imageDrawFaces(self, faces = None, image = None):
		"""Draw all faces and centers from a face dictionary."""
		if not image:
			image = self.image
		
		if not faces:
			faces = self.faces
		
		for face in faces:
			self.imageDrawPolygon(face['draw']['polygon'], image)
			self.imageDrawPoint(face['draw']['center'], image)
		
		return image
	
	
	def hasNearbyFaceCenters(self, center = (0, 0)):
		"""Check to see if a face's center is near a point."""
		
		if self.image.width > self.image.height:
			range_check = int(self.image.width*.015)
		else:
			range_check = int(self.image.height*.015)
			
		center_x, center_y = center
		
		for face in self.faces:
			face_center_x, face_center_y = face['draw']['center']
			
			min_x = face_center_x-range_check
			max_x = face_center_x+range_check
			
			min_y = face_center_y-range_check
			max_y = face_center_y+range_check
			
			in_horizontal_range = min_x < center_x < max_x
			in_vertical_range = min_y < center_y < max_y
			
			if in_horizontal_range and in_vertical_range:
				return True
		
		return False
	
	
	def facesDetect(self):
		"""Create a face dictionary."""
		
		self.faces = []
		classifier_face = self.classifiers['face']
		
		for rotation in [0, 30, -30]:
			classifier_settings = classifier_face['settings'].copy()
			rot_image = self.image.rotate(rotation, resample=self.resample)
			
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
				
				# Rotate these points for drawing on the original image.
				face_polygon = Object2D.rotate(
					face_polygon, rotation, self.center
				)
				
				face_center = Object2D.rotate(
					face_center, rotation, self.center
				)
				
				
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
	
	
	def facesCrop(self, path, faces = None, image = None, normalize = True):
		"""Draw all faces and centers from a face dictionary."""
		
		is_path = isinstance(path, str) and os.path.isfile(path)
		is_img_path = isinstance(self.path, str) and os.path.isfile(self.path)
		
		if not is_path:
			path = False
		else:
			if is_img_path:
				fname = 'unknown'
			else:
				fname = 'unknown'
		
		if not image:
			image = self.image
		
		if not faces:
			faces = self.faces
		
		classifier_left = self.classifiers['eyes_left']
		classifier_right = self.classifiers['eyes_right']
		
		for face in faces:
			rot_image = image.copy()
			
			if face['rotation'] != 0:
				rot_image = rot_image.rotate(
					face['rotation'], resample=self.resample)
			
			if normalize:
# 				crop = list(face['crop'])
# 				
# 				print(crop)
# 				
# 				crop[0] = crop[0]-50
# 				if crop[0] < 0:
# 					crop[0] = 0
# 				
# 				crop[1] = crop[1]-50
# 				if crop[1] < 0:
# 					crop[1] = 0
# 					
# 				crop[2] = crop[2]+50
# 				if crop[2] > rot_image.width:
# 					crop[2] = rot_image.width
# 					
# 				crop[3] = crop[3]+50
# 				if crop[3] > rot_image.height:
# 					crop[3] = rot_image.height
# 				
# 				rot_image = rot_image.crop(crop)
				
				rot_image = rot_image.crop(face['crop'])
				
				dimensions = self.normalized_size
				rot_image = rot_image.resize(dimensions, resample=self.resample)
				
				classifier_settings = classifier_left['settings'].copy()
				found_left = classifier_left['classifier'].detectMultiScale(
					self.imageToCV(rot_image),
					**classifier_settings
				);
				
				classifier_settings = classifier_right['settings'].copy()
				found_right = classifier_left['classifier'].detectMultiScale(
					self.imageToCV(rot_image),
					**classifier_settings
				);
				
				if found_left.any() and found_right.any():
					rot_image = self.imageDrawPolygon(
						Object2D.boxPolygon(found_left),
						rot_image
					)
					
					rot_image = self.imageDrawPolygon(
						Object2D.boxPolygon(found_right),
						rot_image
					)
				
				rot_image.show()
				
			else:
				rot_image = rot_image.crop(face['crop'])
			
			if is_path:
				fname = 'unknown'
			
			
		# Loop
			# Crop Square w Margin
			# Normalize?
				# Detect Eyes
				# Detect Angle
				# Rotate
				# Detect Face
				# Crop
			# Save
	
	def show(self, image = None):
		"""Show an image."""
		if not image:
			self.image.show()
		else:
			image.show()
	
	
		
if __name__ == "__main__":
	# Mia and Erin at Disney
	# img = CVFace('samples/IMG_2492.jpg')
	# Mia, Anna, and Joost
	# img = CVFace('samples/IMG_5493.jpg')
	
	import glob
	
	files = glob.glob('samples/*.jpg')
	
	for file in files:
		img = CVFace(file)
		img.facesCrop('output')
		exit()
		#img.imageDrawFaces()
		#img.show()
		
		pp = pprint.PrettyPrinter(indent=4)
		pp.pprint(file)
		pp.pprint(len(img.faces))
		for face in img.faces:
			if face['box'][2] != face['box'][3]:
				print(face['box'])
		# pp.pprint(img.faces)