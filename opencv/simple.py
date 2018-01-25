from PIL import Image, ImageDraw
import numpy
import cv2
import os

class CvFace:
	def __init__(self, image):
		"""Create image based on type and load cascade classifiers."""
		
		self.classifiers = {
			'eyes': {
				'file': 'haarcascade_eye.xml',
				'settings': {
					'scale': 1.05,
					'neighbor': 6
				},
				'classifier': False,
			},
			'face': {
				'file': 'haarcascade_frontalface_alt2.xml',
				'settings': {
					'scale': 1.05,
					'neighbor': 6
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
	
	
	def loadImageFromPath(self, path):
		image = Image.open(path)
		image_np = numpy.asarray(image)
		image_cv = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
		
		return (image, image_np, image_cv)
	
	
	def loadImageFromPil(self, pil):
		image = pil
		image_np = numpy.asarray(pil)
		image_cv = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
		
		return (image, image_np, image_cv)
	
	
	def loadCascadeClassifiers(self):
		"""Load the classifiers."""
		
		for class_type, class_data in self.classifiers.items():
			if os.path.isfile(class_data['file']):
				classifier = cv2.CascadeClassifier(class_data['file'])
				if not classifier.empty():
					self.classifiers[class_type]['classifier'] = classifier
	
	
	def showRegions(self, regions, image = None):
		"""Draw a box around a region"""
		
		if not image:
			(image, image_np, image_cv) = self.loadImageFromPil(self.image)
		
		draw = ImageDraw.Draw(image)
		for (x,y,w,h) in regions:
			draw.rectangle(((x, y), (x+w, y+h)), outline="blue")
		
		image.show()
		
		return image
		
	
	def getFaces(self):
		"""Starts detection of faces."""
		
		classifier_face = self.classifiers['face']
		classifier_eyes = self.classifiers['eyes']
		faces = []
		rotation = 0
		
		# @todo This needs to keep everything and somehow deduplicate
		# The unrotated should be preferred.
		
		(cur_image, cur_image_np, cur_image_cv) = self.loadImageFromPil(
				self.image
			)
		
		for rotation in [0, 30, -30]:
			(rot_image, rot_image_np, rot_image_cv) = self.loadImageFromPil(
					cur_image.rotate(rotation)
				)
			
			# Rotate 0, 30, and -30 helps find more faces.
			faces_rotated = classifier_face['classifier'].detectMultiScale(
				rot_image_cv, 
				scaleFactor = classifier_face['settings']['scale'], 
				minNeighbors = classifier_face['settings']['neighbor']
			);
			
			if len(faces) < len(faces_rotated):
				faces = faces_rotated
				cur_image_keep = rot_image
				cur_image_np_keep = rot_image_np
				cur_image_cv_keep = rot_image_cv
		
		self.showRegions(faces, cur_image_keep)
		return
		
		
		for face in faces:
			(cur_image, cur_image_np, cur_image_cv) = self.loadImageFromPil(
					self.image
				)
			
			x_padding = 30
			y_padding = 30
			
			cur_image_face_region = [
				face[0]-x_padding, 
				face[1]-y_padding, 
				face[0]+face[2]+x_padding,
				face[1]+face[3]+y_padding
			]
			
			cur_image = cur_image.crop(cur_image_face_region)
			cur_image.load()
			cur_image_np = numpy.asarray(cur_image)
			cur_image_cv = cv2.cvtColor(cur_image_np, cv2.COLOR_BGR2RGB)
			
			eyes = classifier_eyes['classifier'].detectMultiScale(
				cur_image_cv, 
				scaleFactor = classifier_eyes['settings']['scale'], 
				minNeighbors = classifier_eyes['settings']['neighbor']
			);
			
			print(eyes)
			self.showRegions(eyes, cur_image)
			
			for eye in eyes:
				print(eye)
			
			# RETURNS A COPY!
# 			cur_image = cur_image.rotate(90)
# 			cur_image.show()
# 			self.image.show()
			print(face)
		
		# @todo 	Crop with large margin. 
		# @todo 	Crop Beyond width to: 
		# @todo 	max(
		# @todo 		width * cos(rotation) + height * sin(rotation), 
		# @todo 		width * sin(rotation) + height * cos(rotation)
		# @todo 	)
		# @todo 	Detect eyes
		# @todo 	Rotate?
		
		print('OK, I will detect a face once you write the code, OK?')


if __name__ == "__main__":
	face = CvFace('../../Developer/facedetect/faces/IMG_5493.jpg')
	faces = face.getFaces() # Not sure about that