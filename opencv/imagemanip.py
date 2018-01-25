import PIL
import numpy
import cv2

def Image:
	"""Manipulate an image for use in OpenCV"""
	
	def __init__(self, image):
		"""Open the image and prep the PIL and Numpy representations.
		
		This opens an image file depending on what type of parameter 
		is passed. 
		
		If `image` is detected as (something like this):
			string -- Assumes a path.
			numpy.ndarray -- Assumes a numpy image array.
			PIL.JpegImagePlugin.JpegImageFile -- Assumes a PIL image.
		"""
		
		if isinstance(y, str):
			# self.imgpil = PIL.Image.open()
			# self.imgcv = numpy.asarray(self.imgpil)
		elif isinstance(x, np.ndarray):
			# self.imgnp = image
			# self.imgpil = PIL.Image.fromarray(image)
		elif isinstance(img, PIL.JpegImagePlugin.JpegImageFile):
			# self.imgpil = numpy.asarray(self.imgpil)
			# self.imgnp = image
			
		self.update()
	
	def update(self):
		"""Updates the PIL and Numpy representations."""
		# self.imgpil = whatever
		# self.imgnp = whatever
	
	def copy(self):
		"""Returns an instance of the current image.
		
		If you need to rotate and crop an image to crop a face, for 
		example, without reloading the image, just make a copy.
		"""