import cv2
import math
import numpy
import os
import PIL
from PIL import Image, ImageDraw
import time

class CvFace:
	def __init__(self, image):
		"""Create image based on type and load cascade classifiers."""
		
		self.classifiers = {
			'eyes': {
				'file': 'shameem_haarcascade_eye.xml',
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
		
		self.faces = []
		
		self.loadCascadeClassifiers()
		self.loadImage(image)
	
	
	def loadImage(self, image):
		"""Load the image."""
		
		image_is_path = isinstance(image, str) and os.path.isfile(image)
		
		if image_is_path:
			try:
				(image_pil, image_np, image_cv) = self.loadImageFromPath(image)
				self.image = image_pil
				self.image_np = image_np
				self.image_cv = image_cv
			except IOError:
				self.image = False
				self.image_np = False
				self.image_cv = False
		else:
			# @todo Add support for PIL.JpegImagePlugin.JpegImageFile
			# @todo Add support for np.ndarray
			self.image = False
			self.image_np = False
			self.image_cv = False
		
		if self.image:
			width, height = self.image.size
			self.size = (width, height)
			self.center = (width/2, height/2)
	
	
	def loadImageFromPath(self, path):
		"""Load an image from a path."""
		image = Image.open(path)
		image_np = numpy.asarray(image)
		image_cv = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
		
		return (image, image_np, image_cv)
	
	
	def loadImageFromPil(self, pil):
		"""Load the from a PIL object."""
		image = pil.copy()
		image_np = numpy.asarray(pil)
		image_cv = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
		
		return (image, image_np, image_cv)
	
	# @todo This probably needs to be the basis of loadImageFrom* conversions
	def imagePilToCv(self, pil):
		"""Convert PIL to OpenCV."""
		image_np = numpy.asarray(pil)
		image_cv = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
		return image_cv
	
	
	def loadCascadeClassifiers(self):
		"""Load the classifiers."""
		
		for class_type, class_data in self.classifiers.items():
			if os.path.isfile(class_data['file']):
				classifier = cv2.CascadeClassifier(class_data['file'])
				if not classifier.empty():
					self.classifiers[class_type]['classifier'] = classifier
	
	
	def rotatePoints(self, origin, point, angle):
		"""Rotate a point by a given angle around a given origin."""
		
		# Taken from https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
		
		angle = math.radians(angle)
		
		ox, oy = origin
		px, py = point
		
		qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
		qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
		
		return (qx, qy)
	
	
	def showRegions(self, regions, image = None, shape = 'square', show = False):
		"""Draw a box around a region"""
		
		colors = ['Yellow', 'Magenta', 'Teal', 'DeepPink', 'Gold', 
			'MediumSlateBlue', 'Blue', 'Red', 'MediumSpringGreen', 'Green', 
			'Cyan', 'Brown', 'Orange']
		
		color_count = 0
		
		if not image:
			(image, image_np, image_cv) = self.loadImageFromPil(self.image)
		
		draw = ImageDraw.Draw(image)
		if shape == 'square':
			for (x1,y1,x2,y2) in regions:
				draw.rectangle(
					[(x1, y1), (x1+x2, y1+y2)], 
					outline=colors[color_count]
				)
				color_count = color_count+1
				if color_count > len(colors):
					color_count = 0
		else:
			for region in regions:
				draw.polygon(region, outline=colors[color_count])
				color_count = color_count+1
				if color_count > len(colors):
					color_count = 0
		
		if show:
			image.show()
		
		return image
	
	def showFaces(self, show = False):
		colors = ['Yellow', 'Magenta', 'Teal', 'DeepPink', 'Gold', 
			'MediumSlateBlue', 'Blue', 'Red', 'MediumSpringGreen', 'Green', 
			'Cyan', 'Brown', 'Orange']
		
		color_count = 0
		
		tmp_img = self.image.copy()
		draw = ImageDraw.Draw(tmp_img)
		for face in self.faces:
			draw.polygon(face['viewpoly'], outline=colors[color_count])
			draw.ellipse(((face['viewcenter'][0]-3, face['viewcenter'][1]-3), (face['viewcenter'][0]+6, face['viewcenter'][1]+6)), fill=colors[color_count])
			color_count = color_count+1
			if color_count >= len(colors):
				color_count = 0
		
		if show:
			tmp_img.show()
		
		return tmp_img
		
	
	def getFaces(self, requireEyes = False):
		"""Get details about faces in the image."""
		
		classifier_face = self.classifiers['face']
		classifier_eyes = self.classifiers['eyes']
		
		faces = []
		
		for rotation in [0, 30, -30]:
			rot_image = self.image.rotate(rotation, resample=PIL.Image.BILINEAR)
			rot_image_cv = self.imagePilToCv(rot_image)
			
			found_faces = classifier_face['classifier'].detectMultiScale(
				rot_image_cv, 
				**classifier_face['settings']
			);
			
			for (top, left, width, height) in found_faces:
				right = left + width
				bottom = top + height
				center = (top + width/2, left + height/2)
				
				polybox = [
					(top, left),
					(top, right),
					(bottom, right),
					(bottom, left)
				]
				
				nat_cent_x, nat_cent_y = self.rotatePoints(self.center, 
					center, rotation)
				
				center = (nat_cent_x, nat_cent_y)
				
				use_rotated_face = True
				
				if rotation != 0:
					
					for face in faces:
						# Seems around 1.5% of the max dimension works to
						# find faces detected sideways that were also detected
						# in normal mode. It doesn't work every single time
						# but it's pretty close.
						if self.image.size[0] > self.image.size[1]:
							range_check = int(self.image.size[0]*.015)
						else:
							range_check = int(self.image.size[1]*.015)
						
						test_cent_x = face['viewcenter'][0]
						test_cent_y = face['viewcenter'][1]
						near_x = (test_cent_x-range_check < nat_cent_x < test_cent_x+range_check)
						near_y = (test_cent_y-range_check < nat_cent_y < test_cent_y+range_check)
						
						if near_x and near_y:
							use_rotated_face = False
				
				if use_rotated_face:
					face = {
						'rotation': rotation,
						'eyes': False,
						'eyerotation': False,
						'crop': (top, left, bottom, right),
						'box': (top, left, width, height),
						'viewcenter': center,
						'viewpoly': []
					}
					
					for point in polybox:
						face['viewpoly'].append(
							self.rotatePoints(self.center, point, rotation)
						)
					
					if requireEyes:
						rot_image_crop = rot_image.copy()
						rot_image_crop = rot_image_crop.crop(face['crop'])
						
						found_eyes = classifier_eyes['classifier'].detectMultiScale(
							self.imagePilToCv(rot_image_crop), 
							**classifier_eyes['settings']
						);
						
						self.showRegions(found_eyes, rot_image_crop, 'square', True)
					
					faces.append(face)
			
		self.faces = faces
		return faces
			


if __name__ == "__main__":
	import glob
	import os
	
	files = glob.glob('samples/*.jpg')
	
	for file in files:
		face = CvFace(file)
		face.getFaces(True)
# 		face.cropFaces('output/')
# 		tmp_img = face.showFaces()
# 		tmp_img.save(file.replace('samples/', 'output/'))