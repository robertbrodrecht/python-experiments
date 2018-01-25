import cv2
import math
import numpy
import os
from PIL import Image, ImageDraw

def rotatePoints(origin, point, angle):
	"""Rotate a point by a given angle around a given origin."""
	
	# Taken from https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
	
	angle = math.radians(angle)
	
	ox, oy = origin
	px, py = point
	
	qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
	qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
	
	return qx, qy

image = Image.open('rot.jpg')

imw,imh = image.size
center = (imw/2, imh/2)

w = 100
h = 100
t = 100
l = imw/2-w/2
b = t+h
r = l+w

sqpoly = [(t,l), (t,r), (b,r), (b,l)]

draw = ImageDraw.Draw(image)
#draw.polygon(sqpoly, outline='red')

for rot in range(20, 360, 30):
	sqpoly_rot = []
	for point in sqpoly:
		rotx,roty = rotatePoints(center, point, rot)
		sqpoly_rot.append((rotx, roty))
	draw.polygon(sqpoly_rot, outline='white')

image.show()