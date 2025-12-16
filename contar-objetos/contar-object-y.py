import cv2
import numpy as np

print('libreria cv2 cargada correctamente')

img = cv2.imread('Cinta-Hilos.jpg')
img = cv2.resize(img, (int((2643/5)), int((2069/5))))

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()