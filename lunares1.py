#from _typeshed import Self
import sys
from PyQt5 import QtWidgets, uic
from PyQt5.sip import delete
#from PyQt5.uic import loadUi
#from PyQt5 import QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QTextEdit, QWidget, QMainWindow, QListWidgetItem,QListWidget, QFileDialog, QComboBox, QMessageBox, QTabWidget, QAction
from PyQt5.QtGui import QIcon, QPixmap, QImage, QBitmap
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import scipy as sp
from scipy import signal ,io
from scipy.io import loadmat
import math as m
import SimpleITK as sitk
import cv2
from PIL import Image
import ctypes
#import skimage
#import shapely
from skimage import morphology
from shapely.geometry import LineString

class INICIO(QMainWindow):
	def __init__(self):
		super(INICIO, self).__init__()
		uic.loadUi("interfaz2.ui", self)

		
  
		#self.resize(2500, 1500)

		self.show()
		self.MainWindow = self.findChild(QMainWindow, "MainWindow")

		self.cargar_imagen.clicked.connect(self.f_cargar_imagen)
		self.boton_segmentar.clicked.connect(self.f_boton_segmentar)
		self.boton_dilatacion.clicked.connect(self.f_boton_dilatacion)
		self.boton_erosion.clicked.connect(self.f_boton_erosion)
		self.boton_apertura.clicked.connect(self.f_boton_apertura)
		self.boton_cierre.clicked.connect(self.f_boton_cierre)
  
		self.boton_diagnostico.clicked.connect(self.diagnostico)
  
		self.boton_ayuda.clicked.connect(self.f_ayuda)

	def f_cargar_imagen(self):
		
		self.imagen_original = []
		self.coord_recorte = [0, 0, 0, 0]
		self.imagen_recortada = []
		self.imagen_recortada_print = []
		self.correcciones = [0, 0, 0, 0]
		self.imagen_segmentada = []
		self.imagen_segmentada_mejorada = []
		self.imagen_perimetro = []
		self.path = ''
		self.diam_Max = 0

		self.cargar_x0.clear()
		self.cargar_x.clear()
		self.cargar_y0.clear()
		self.cargar_y.clear()
  
		self.boton_dilatacion.setEnabled(False)
		self.boton_erosion.setEnabled(False)
		self.boton_apertura.setEnabled(False)
		self.boton_cierre.setEnabled(False)
		self.boton_diagnostico.setEnabled(False)
  
		self.label_indice_A.setText('')
		self.label_indice_B.setText('')
		self.label_indice_C.setText('')
		self.label_indice_D.setText('')
		self.label_TDS.setText('')
		self.label_diagnostico.setText('')
  
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		self.path, _ = QFileDialog.getOpenFileName(None,"Seleccione su Lunar","","Images (*.png *.xpm *.jpg *.jpeg *.bmp)")
		
		if self.path != (''):
			originalPrintear = QPixmap(self.path)
			original = cv2.imread(self.path, 0)
			self.imagen_original = np.zeros([original.shape[0], original.shape[1]])
			self.imagen_original = original

			originalPrintear2 = originalPrintear.scaled(351, 271)
			self.imagen_in.setPixmap(originalPrintear2)
		
			filas = np.shape(self.imagen_original)[0]
			cols = np.shape(self.imagen_original)[1]
			self.eje_y0_in.setText("0")
			self.eje_y_in.setText(str(filas))
			self.eje_x0_in.setText("0")
			self.eje_x_in.setText(str(cols))
   
			self.boton_segmentar.setEnabled(True)
			self.cargar_x0.setEnabled(True)
			self.cargar_x.setEnabled(True)
			self.cargar_y0.setEnabled(True)
			self.cargar_y.setEnabled(True)
   
			self.txt_x0.setEnabled(True)
			self.txt_x.setEnabled(True)
			self.txt_y0.setEnabled(True)
			self.txt_y.setEnabled(True)
		else:
			ctypes.windll.user32.MessageBoxW(0, "No cargó la imagen.", "Error", 1)


	def f_boton_segmentar(self):
		
		if self.cargar_x0.toPlainText().isnumeric() and self.cargar_x.toPlainText().isnumeric() and self.cargar_y0.toPlainText().isnumeric() and self.cargar_y.toPlainText().isnumeric():
			if (int(self.cargar_x0.toPlainText()) < int(self.cargar_x.toPlainText())) and (int(self.cargar_y0.toPlainText()) < int(self.cargar_y.toPlainText())):
				if ( int(self.cargar_x.toPlainText()) < self.imagen_original.shape[1] ) and ( int(self.cargar_y.toPlainText()) < self.imagen_original.shape[0] ):
					self.coord_recorte[0] = int(self.cargar_x0.toPlainText())
					self.coord_recorte[1] = int(self.cargar_x.toPlainText())
					self.coord_recorte[2] = int(self.cargar_y0.toPlainText())
					self.coord_recorte[3] = int(self.cargar_y.toPlainText())
    
					self.recorte_manual()
					self.imagen_segmentada = np.zeros([self.imagen_recortada.shape[0], self.imagen_recortada.shape[1]])
					_, self.imagen_segmentada = cv2.threshold(self.imagen_recortada, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
					self.correccion_margen()
					self.imagen_segmentada_mejorada = self.imagen_segmentada.copy()
    
					#height, width = self.imagen_recortada_print.shape
					#bytesPerLine = 3 * width
					#b = (255 << 24 | a[:,:,0] << 16 | a[:,:,1] << 8 | a[:,:,2]).flatten()
					#self.imagen_recortada_print = QImage(self.imagen_recortada_print, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
    
					#self.imagen_recortada_print = Image.fromarray(self.imagen_recortada_print)
					#self.imagen_recortada_print = self.imagen_recortada_print.scaled(351, 271)
					#self.imagen_in.setPixmap(self.imagen_recortada_print)
		
					cv2.destroyAllWindows() 
					cv2.imshow("Imagen Recortada", self.imagen_recortada)
					cv2.imshow("Imagen Segmentada", self.imagen_segmentada)
    
					self.boton_dilatacion.setEnabled(True)
					self.boton_erosion.setEnabled(True)
					self.boton_apertura.setEnabled(True)
					self.boton_cierre.setEnabled(True)
					self.boton_diagnostico.setEnabled(True)
  
					#filas_out = np.shape(self.imagen_recortada_print)[0]
					#cols_out = np.shape(self.imagen_recortada_print)[1]
					#self.eje_y0_out.setText("0")
					#self.eje_y_out.setText(str(filas_out))
					#self.eje_x0_out.setText("0")
					#self.eje_x_out.setText(str(cols_out))
	
				elif int(self.cargar_x.toPlainText()) > self.imagen_original.shape[1] :
					ctypes.windll.user32.MessageBoxW(0, "El valor X no puede superar el tamaño horizontal de la imagen. Vuelva a intentar", "Error", 1)
     
				elif int(self.cargar_y.toPlainText()) > self.imagen_original.shape[0] :
					ctypes.windll.user32.MessageBoxW(0, "El valor Y no puede superar el tamaño vertical de la imagen. Vuelva a intentar", "Error", 1)
   
			elif(int(self.cargar_x0.text()) >= int(self.cargar_x.text())):
				ctypes.windll.user32.MessageBoxW(0, "El valor X0 no puede ser mayor o igual que X. Vuelva a intentar", "Error", 1)

			elif(int(self.cargar_y0.text()) >= int(self.cargar_y.text())):
				ctypes.windll.user32.MessageBoxW(0, "El valor Y0 no puede ser mayor o igual que Y. Vuelva a intentar", "Error", 1)

		else:
			ctypes.windll.user32.MessageBoxW(0, "Se ha introducido una o más variables no numéricas. Vuelva a intentar", "Error", 1)

	def recorte_manual(self):
		x_min = self.coord_recorte[0]
		x_max = self.coord_recorte[1]
		y_min = self.coord_recorte[2]
		y_max = self.coord_recorte[3]
		
		self.imagen_recortada = np.zeros([self.imagen_original.shape[0], self.imagen_original.shape[1]])
		#self.imagen_recortada_print = np.zeros([self.imagen_original.shape[0], self.imagen_original.shape[1]])
		self.imagen_recortada = self.imagen_original[y_min:y_max, x_min:x_max]
		#self.imagen_recortada_print = self.imagen_original[y_min:y_max,x_min:x_max]
  
		if (x_min == 0):
			self.correcciones[0] = 10
  
		if (x_max == self.imagen_original.shape[1]-1):
			self.correcciones[1] = 10

		if (y_min == 0):
			self.correcciones[2] = 10

		if (y_max == self.imagen_original.shape[0]-1):
			self.correcciones[3] = 10
  
	def correccion_margen(self):
		M = len(self.imagen_segmentada)               #filas imagen
		N = len(self.imagen_segmentada[0])            #cols imagen

		img_corregida = np.zeros([M, N])
		print('Dimensiones imagen_segmentada: ', M, ' y ', N)
		img_corregida[self.correcciones[2]:M-self.correcciones[3], self.correcciones[0]:N-self.correcciones[1]] = self.imagen_segmentada[self.correcciones[2]:M-self.correcciones[3], self.correcciones[0]:N-self.correcciones[1]]

		img_out = np.zeros([M+self.correcciones[2]+self.correcciones[3],N+self.correcciones[0]+self.correcciones[1]])
 
		M2 = len(img_out)               #filas imagen
		N2 = len(img_out[0])
		print('Dimensiones imagen_corregida: ', M2, ' y ', N2)
		print('Correcciones: ', self.correcciones)
		print(img_out.shape)
		print(img_corregida.shape)
		img_out[self.correcciones[2]:M2-self.correcciones[3], self.correcciones[0]:N2-self.correcciones[1]] = img_corregida

		self.imagen_segmentada = img_out

	###################################################################
	# OPERACIONES MORFOLOGICAS

	def f_boton_dilatacion(self):
		kernel = np.ones((5, 5), 'uint8')
		self.imagen_segmentada_mejorada = cv2.dilate(self.imagen_segmentada_mejorada, kernel, iterations=1)
		cv2.destroyAllWindows() 
		cv2.imshow("Imagen Recortada", self.imagen_recortada)
		cv2.imshow("Imagen Segmentada", self.imagen_segmentada)
		cv2.imshow("Imagen Segmentada Mejorada", self.imagen_segmentada_mejorada)

	def f_boton_erosion(self):
		kernel = np.ones((5, 5), 'uint8')
		self.imagen_segmentada_mejorada = cv2.erode(self.imagen_segmentada_mejorada, kernel, iterations=1)
		cv2.destroyAllWindows() 
		cv2.imshow("Imagen Recortada", self.imagen_recortada)
		cv2.imshow("Imagen Segmentada", self.imagen_segmentada)
		cv2.imshow("Imagen Segmentada Mejorada", self.imagen_segmentada_mejorada)
  
	def f_boton_apertura(self):
		kernel = np.ones((5, 5), 'uint8')
		aux = cv2.erode(self.imagen_segmentada_mejorada, kernel, iterations=1)
		self.imagen_segmentada_mejorada = cv2.dilate(aux, kernel, iterations=1)
		cv2.destroyAllWindows() 
		cv2.imshow("Imagen Recortada", self.imagen_recortada)
		cv2.imshow("Imagen Segmentada", self.imagen_segmentada)
		cv2.imshow("Imagen Segmentada Mejorada", self.imagen_segmentada_mejorada)

	def f_boton_cierre(self):
		kernel = np.ones((5, 5), 'uint8')
		aux = cv2.dilate(self.imagen_segmentada_mejorada, kernel, iterations=1)
		self.imagen_segmentada_mejorada = cv2.erode(aux, kernel, iterations=1)
		cv2.destroyAllWindows() 
		cv2.imshow("Imagen Recortada", self.imagen_recortada)
		cv2.imshow("Imagen Segmentada", self.imagen_segmentada)
		cv2.imshow("Imagen Segmentada Mejorada", self.imagen_segmentada_mejorada)	
 
	###################################################################
	# PARAMETROS
 
	def area(self, imagen):  # Revisar
		cant = 0
		for i in range(imagen.shape[0]):
			for j in range(imagen.shape[1]):

				if (imagen[i,j] == 255):
					cant += 1

		return cant
		
	def perimetro(self):  # Revisar
		self.imagen_perimetro = np.zeros((self.imagen_segmentada_mejorada.shape[0],self.imagen_segmentada_mejorada.shape[1]))

		for i in range(self.imagen_segmentada_mejorada.shape[0]):

			interior = 0
			for j in range(self.imagen_segmentada_mejorada.shape[1]):

				if (self.imagen_segmentada_mejorada[i,j]) == 255 and (interior == 0):
					self.imagen_perimetro[i,j] = 255
					interior = 1
				elif (self.imagen_segmentada_mejorada[i,j]) == 0 and (interior == 1):
					self.imagen_perimetro[i,j-1] = 255
					interior = 0

		for j in range(self.imagen_segmentada_mejorada.shape[1]):

			interior = 0
			for i in range(self.imagen_segmentada_mejorada.shape[0]):

				if (self.imagen_segmentada_mejorada[i,j]) == 255 and (interior == 0):
					self.imagen_perimetro[i,j] = 255
					interior = 1
				elif (self.imagen_segmentada_mejorada[i,j]) == 0 and (interior == 1):
					self.imagen_perimetro[i-1,j] = 255
					interior = 0

		cant = 0

		for i in range(self.imagen_perimetro.shape[0]):
			for j in range(self.imagen_perimetro.shape[1]):
      
				if (self.imagen_perimetro[i,j] == 255):
					cant += 1

		return cant

	def centroide(self, imagen):
		xc = 0
		yc = 0

		for i in range(imagen.shape[0]):
			for j in range(imagen.shape[1]):
				if(imagen[i,j] == 255):
					xc += j
					yc += i
		#print("paso previo del centroide es: ", xc, ' y ', yc)
		area = self.area(self.imagen_segmentada_mejorada)
		perimetro = self.perimetro()   # Perimetro de imagen_segmentada_mejorada
		xc = round(xc/(area-perimetro))
		yc = round(yc/(area-perimetro))
		#print("El area es: ", area)
		#print("El perimetro es: ", perimetro)
		#print("El centroide es: ", xc, ' y ', yc)
		return xc, yc

	def longest_diagonal_distance(self):   # Revisar
		_ = self.perimetro()
		coordenadas_X = []
		coordenadas_Y = []

		for i in range(self.imagen_perimetro.shape[0]):
			for j in range(self.imagen_perimetro.shape[1]):
				if(self.imagen_perimetro[i,j] == 255):
					coordenadas_Y.append(i)
					coordenadas_X.append(j)
     
		x1 = 0
		x2 = 0
		y1 = 0
		y2 = 0
		for i in range(len(coordenadas_X)-1):
			for j in range(i+1,len(coordenadas_X)):
				diam = m.sqrt((coordenadas_X[i]-coordenadas_X[j])**2 + (coordenadas_Y[i]-coordenadas_Y[j])**2)
				if (diam>self.diam_Max):
					self.diam_Max = diam
					x1 = coordenadas_X[i]
					x2 = coordenadas_X[j]
					y1 = coordenadas_Y[i]
					y2 = coordenadas_Y[j]
  
		return x1, x2, y1, y2
  
	##############################################################
	# Indices
	def asimetria_ejes(self, x1, x2, y1, y2):
		imagen_ejes = self.imagen_perimetro.copy()

		a = [x1, y1]
		b = [x2, y2]
		cd_length = 2000

		punto_medio = [round((x1 + x2)/2), round((y1 + y2)/2)]      # columna, fila

		ab = LineString([a, punto_medio])
		left = ab.parallel_offset(cd_length / 2, 'left')
		right = ab.parallel_offset(cd_length / 2, 'right')
		c = left.boundary[1]
		d = right.boundary[0]  # note the different orientation for right offset

		cd = LineString([c, punto_medio])
		cv2.line(imagen_ejes, (int(c.x), int(c.y)), (int(d.x), int(d.y)), (200, 200, 200), 1)

		left = cd.parallel_offset(cd_length / 2, 'left')
		right = cd.parallel_offset(cd_length / 2, 'right')
		e = left.boundary[1]
		f = right.boundary[0]  # note the different orientation for right offset

		cv2.line(imagen_ejes, (int(e.x), int(e.y)), (int(f.x), int(f.y)), (100, 100, 100), 1)
  
		imagen_eje1_area = self.imagen_segmentada_mejorada.copy()
		imagen_eje2_area = self.imagen_segmentada_mejorada.copy()

		cv2.line(imagen_eje2_area, (int(c.x), int(c.y)), (int(d.x), int(d.y)), (200, 200, 200), 1)
		cv2.line(imagen_eje1_area, (int(e.x), int(e.y)), (int(f.x), int(f.y)), (100, 100, 100), 1)   # Principal
	
		return imagen_eje1_area, imagen_eje2_area
 
	def asimetria(self, imagen, color_eje):
		area1 = 0
		area2 = 0
		H_sup = -1
		H_inf = -1
		V_izq = -1
		V_der = -1

		#Recorremos los bordes para ver qué caso estamos trabajando:
		for i in range(imagen.shape[0]):
			if imagen[i, 0] == color_eje:
				V_izq = i
			if imagen[i, -1] == color_eje:
				V_der = i
  
		for j in range(imagen.shape[1]):
			if imagen[0, j] == color_eje:
				H_sup = j
			if imagen[-1, j] == color_eje:
				H_inf = j

		#Ejecuto el calculo de areas
		if (H_sup != -1 and H_inf != -1): #Caso 1: eje entre el margen superior y el inferior
			for i in range(imagen.shape[0]):
				mitad = 0
				for j in range(imagen.shape[1]):
					if imagen[i, j] == color_eje:
						mitad = 1
					elif imagen[i, j] == 255 and mitad == 0:
						area1 += 1
					elif imagen[i, j] == 255 and mitad == 1:
						area2 += 1
  
		elif (V_izq != -1 and V_der != -1): #Caso 2: eje entre el margen izquierdo y derecho 
			for j in range(imagen.shape[1]):
				mitad = 0
				for i in range(imagen.shape[0]):
					if imagen[i, j] == color_eje:
						mitad = 1
					elif imagen[i, j] == 255 and mitad == 0:
						area1 += 1
					elif imagen[i, j] == 255 and mitad == 1:
						area2 += 1
		else:
			print('Caso no contemplado')
		print("Las areas son: ", area1, ' y ', area2)
		a_dif = (abs(area1-area2)*100/(area1+area2))
		indice = 0
		if a_dif > 17.5:  # Estaba en 17
			indice += 1
		print('a_dif es ', a_dif)
		return indice
  
	def bordes_segmentos(self):
		img_bordes = self.imagen_perimetro.copy()
		#----- PASO 1: Divido en 8 segementos, trazando rectas verticales, horizontales y diagonales pasantes por el centroide -----
		# Centroide
		xc, yc = self.centroide(self.imagen_segmentada_mejorada)  # estaba en img_bordes
  
		#Vertical:
		for i in range(img_bordes.shape[0]):
			img_bordes[i,xc] = 128

		#Horizontal
		for j in range(img_bordes.shape[1]):
			img_bordes[yc,j] = 128

		#Diagonal abajo-izq hacia arriba-der:
		i = yc
		j = xc
		while (i<img_bordes.shape[0] and j>=0):
			img_bordes[i,j] = 128
			i += 1
			j -= 1

		i = yc
		j = xc
		while (i>=0 and j<img_bordes.shape[1]):
			img_bordes[i,j] = 128
			i -= 1
			j +=1

		#Diagonal arriba-izq hacia abajo-der:
		i = yc
		j = xc
		while (i>=0 and j>=0):
			img_bordes[i,j] = 128
			i -= 1
			j -=1

		i = yc
		j = xc
		while (i<img_bordes.shape[0] and j<img_bordes.shape[1]):
			img_bordes[i,j] = 128
			i += 1
			j += 1
     
		#Llamo seccion 1 a la mitad inferior del cuadrante superior izqueirdo, y luego recorro en sentido horario:
		img_bordes1 = np.zeros((img_bordes.shape[0],img_bordes.shape[1]))
		img_bordes2 = np.zeros((img_bordes.shape[0],img_bordes.shape[1]))
		img_bordes3 = np.zeros((img_bordes.shape[0],img_bordes.shape[1]))
		img_bordes4 = np.zeros((img_bordes.shape[0],img_bordes.shape[1]))
		img_bordes5 = np.zeros((img_bordes.shape[0],img_bordes.shape[1]))
		img_bordes6 = np.zeros((img_bordes.shape[0],img_bordes.shape[1]))
		img_bordes7 = np.zeros((img_bordes.shape[0],img_bordes.shape[1]))
		img_bordes8 = np.zeros((img_bordes.shape[0],img_bordes.shape[1]))
     
		#Seccion 1:
		for j in range(xc):
  
			i = yc - 1
			while (i > -1 and img_bordes[i,j] != 128):
				if (img_bordes[i,j] == 255):
					img_bordes1[i,j] = 255

				i -= 1
    
		#Seccion 2:
		for i in range(yc):
  
			j = xc-1
			while (j > -1 and img_bordes[i,j] != 128):   
				if (img_bordes[i,j] == 255):
					img_bordes2[i,j] = 255
    
				j -= 1

		#Seccion 3:
		for i in range(yc):
  
			j = xc+1
			while (j < img_bordes.shape[1] and img_bordes[i,j] != 128):
				if (img_bordes[i,j] == 255):
					img_bordes3[i,j] = 255
    
				j += 1

		#Seccion 4:
		for j in reversed(range(xc,img_bordes.shape[1])):
  
			i = yc - 1
			while (i > -1 and img_bordes[i,j] != 128):
				if (img_bordes[i,j] == 255):
					img_bordes4[i,j] = 255
    
				i -= 1

		#Seccion 5:
		for j in reversed(range(xc,img_bordes.shape[1])):
  
			i = yc + 1
			while (i < img_bordes.shape[0] and img_bordes[i,j] != 128):
				if (img_bordes[i,j] == 255):
					img_bordes5[i,j] = 255
    
				i += 1

		#Seccion 6:
		for i in reversed(range(yc+1,img_bordes.shape[0])):
  
			j = xc + 1
			while (j < img_bordes.shape[1] and img_bordes[i,j] != 128):
				if (img_bordes[i,j] == 255):
					img_bordes6[i,j] = 255
    
				j += 1

		#Seccion 7:
		for i in reversed(range(yc+1,img_bordes.shape[0])):
  
			j = xc - 1
			while (j > -1 and img_bordes[i,j] != 128):
				if (img_bordes[i,j] == 255):
					img_bordes7[i,j] = 255

				j -= 1

		#Seccion 8:
		for j in range(xc):
  
			i = yc + 1
			while (i < img_bordes.shape[0] and img_bordes[i,j] != 128):
				if (img_bordes[i,j] == 255):
					img_bordes8[i,j] = 255
    
				i += 1

		cv_bordes = [0, 0, 0, 0, 0, 0, 0, 0]
		cv_bordes[0] = self.distancia_al_borde_segmento(img_bordes1, xc, yc)
		cv_bordes[1] = self.distancia_al_borde_segmento(img_bordes2, xc, yc)
		cv_bordes[2] = self.distancia_al_borde_segmento(img_bordes3, xc, yc)
		cv_bordes[3] = self.distancia_al_borde_segmento(img_bordes4, xc, yc)
		cv_bordes[4] = self.distancia_al_borde_segmento(img_bordes5, xc, yc)
		cv_bordes[5] = self.distancia_al_borde_segmento(img_bordes6, xc, yc)
		cv_bordes[6] = self.distancia_al_borde_segmento(img_bordes7, xc, yc)
		cv_bordes[7] = self.distancia_al_borde_segmento(img_bordes8, xc, yc)

		indice = 0
		for i in range(8):
			print("El coef de var es: ", cv_bordes[i])
			if cv_bordes[i] > 0.06:  # 0.065
				indice += 1
  
		return indice


	def distancia_al_borde_segmento(self, imagen, xc, yc):
		# Centroide
		#x_c, y_c = self.centroide(self.imagen_segmentada_mejorada)  # estaba en imagen_perimetro

		distancias = []

		for i in range(imagen.shape[0]):
			for j in range(imagen.shape[1]):
				if (imagen[i,j] == 255):
					d = m.sqrt((j-xc)**2 + (i-yc)**2)
					distancias.append(d)

		desvio = np.std(distancias)
		media = np.mean(distancias)
		varianza = np.var(distancias)
		CV = desvio/media

		return CV

	def color(self):
		imagen2 = cv2.imread(self.path)
		imagen2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2RGB)

		x_min = self.coord_recorte[0]
		x_max = self.coord_recorte[1]
		y_min = self.coord_recorte[2]
		y_max = self.coord_recorte[3]
		
		imagen2 = imagen2[y_min:y_max, x_min:x_max]

		cantidad = 0
		for i in range(0,imagen2.shape[0]):
			for j in range(0,imagen2.shape[1]):
				if (self.imagen_segmentada_mejorada[i,j] == 255): #Solo píxeles del lunar, calculo la distancia de su nivel de gris a los de referencia
					cantidad+=1

		R =imagen2[:,:,0]
		G =imagen2[:,:,1]
		B= imagen2[:,:,2]

		count_white = 0
		count_black = 0
		count_red = 0
		count_lightbrown = 0
		count_darkbrown = 0
		count_bluegray = 0

		cantidad_umbral_white = 0.05 * cantidad 
		cantidad_umbral_black = 0.05 * cantidad 
		cantidad_umbral_red = 0.03 * cantidad 
		cantidad_umbral_lightbrown = 0.225 * cantidad 
		cantidad_umbral_darkbrown = 0.225 * cantidad 
		cantidad_umbral_bluegray = 0.55 * cantidad 

		r_white = [255,255,255]
		r_black = [0,0,0]
		r_red = [255,0,0]
		r_lightbrown = [200,150,100]
		r_darkbrown = [100,50,0]
		r_bluegray = [100,125,150]

		T_white = round(255*np.sqrt(3*(1-0.8039)**2))
		T_black = round(255*np.sqrt(3*(0-0.1961)**2))
		T_red = round(255*np.sqrt(2*(0-0.1961)**2 + (1-0.5882)**2)) 
		T_lightbrown = round(255*np.sqrt((0.7843-0.5882)**2 + (0.5882-0.1961)**2 + (0.3922)**2))
		T_darkbrown = round(255*np.sqrt((0.5882-0.1961)**2 + 2*(0.3922-0)**2))
		T_bluegray = round(255*np.sqrt((0.5882)**2 + (0.4902-0.3922)**2))

		w = 0
		b = 0
		r = 0
		lb = 0
		db = 0
		bg = 0

		for i in range(0,imagen2.shape[0]):
			for j in range(0,imagen2.shape[1]):
				if (self.imagen_segmentada_mejorada[i,j] == 255): #Solo píxeles del lunar, calculo la distancia de su nivel de gris a los de referencia
					D_k_white = m.sqrt((R[i,j] - r_white[0])**2 + (G[i,j] - r_white[1])**2 + (B[i,j] - r_white[2])**2)
					D_k_black = m.sqrt((R[i,j] - r_black[0])**2 + (G[i,j] - r_black[1])**2 + (B[i,j] - r_black[2])**2)
					D_k_red = m.sqrt((R[i,j] - r_red[0])**2 + (G[i,j] - r_red[1])**2 + (B[i,j] - r_red[2])**2)
					D_k_lightbrown = m.sqrt((R[i,j] - r_lightbrown[0])**2 + (G[i,j] - r_lightbrown[1])**2 + (B[i,j] - r_lightbrown[2])**2)
					D_k_darkbrown = m.sqrt((R[i,j] - r_darkbrown[0])**2 + (G[i,j] - r_darkbrown[1])**2 + (B[i,j] - r_darkbrown[2])**2)
					D_k_bluegray = m.sqrt((R[i,j] - r_bluegray[0])**2 + (G[i,j] - r_bluegray[1])**2 + (B[i,j] - r_bluegray[2])**2)

					if (D_k_white <= T_white and w == 0):
						count_white += 1
						if (count_white >= cantidad_umbral_white):
							w = 1
							print("Hay White")

					elif (D_k_black <= T_black and b == 0):
						count_black += 1
						if (count_black >= cantidad_umbral_black):
							b = 1
							print("Hay Black")

					elif (D_k_red <= T_red and r == 0):
						count_red += 1
						if (count_red >= cantidad_umbral_red):
							r = 1
							print("Hay Red")

					elif (D_k_lightbrown <= T_lightbrown and lb == 0):
						count_lightbrown += 1
						if (count_lightbrown >= cantidad_umbral_lightbrown):
							lb = 1
							print("Hay Light Brown")
        
					elif (D_k_darkbrown <= T_darkbrown and db == 0):
						count_darkbrown += 1
						if (count_darkbrown >= cantidad_umbral_darkbrown):
							db = 1
							print("Hay Dark Brown")

					elif (D_k_bluegray <= T_bluegray and bg == 0):
						count_bluegray += 1
						if (count_bluegray >= cantidad_umbral_bluegray):
							bg = 1
							print("Hay Blue Gray")

		C = w + b + r + lb + db + bg
		return C

	def diametro(self):
		dpi = 96 #El del dermatoscopio de la base de datos
		M = self.diam_Max*25.4/(20*dpi)
		if (M< 1):
			D = 0
		elif (M< 2):
			D = 0.5
		elif (M< 3):
			D = 1
		elif (M< 4):
			D = 1.5
		elif (M< 5):
			D = 2
		elif (M< 6):
			D = 2.5
		elif (M< 7):
			D = 3
		elif (M< 8):
			D = 3.5
		elif (M< 9):
			D = 4
		elif (M< 10):
			D = 4.5
		else:
			D = 5

		return D

	def diagnostico(self):
		self.perimetro()
		x1, x2, y1, y2 = self.longest_diagonal_distance()
  
		# Indice Asimetria
		imagen_eje1, imagen_eje2 = self.asimetria_ejes(x1, x2, y1, y2)
		print("shape imagen eje 1: ", imagen_eje1.shape)
		print("shape imagen eje 1: ", imagen_eje2.shape)
		aux1_asimetria = self.asimetria(imagen_eje1, 100)
		aux2_asimetria = self.asimetria(imagen_eje2, 200)
		indice_asimetria = aux1_asimetria + aux2_asimetria
		print("Indices de asimetria: ", aux1_asimetria, ' y ', aux2_asimetria)
		#print(indice_asimetria)	
		#cv2.destroyAllWindows()
		#cv2.imshow("Imagen eje1", imagen_eje1)
		#cv2.imshow("Imagen eje2", imagen_eje2)
		#cv2.imshow("Imagen Segmentada Mejorada", self.imagen_segmentada_mejorada)

		# Indice Borde
		indice_borde = self.bordes_segmentos()
		#print(indice_borde)

		# Indice Color
		indice_color = self.color()
		#print(indice_color)
	
		# Indice Diametro
		indice_diametro = self.diametro()
		#print(indice_diametro)
 
		TDS = 1.3*indice_asimetria +  0.1*indice_borde + 0.5*indice_color + 0.5*indice_diametro
		TDS = round(TDS, 1)

		self.label_indice_A.setText('Asimetría: ' + str(indice_asimetria))
		self.label_indice_B.setText('Bordes: ' + str(indice_borde))
		self.label_indice_C.setText('Color: ' + str(indice_color))
		self.label_indice_D.setText('Diámetro: ' + str(indice_diametro))
		self.label_TDS.setText('TDS: ' + str(TDS))
		
		if (TDS < 5.65):
			self.label_diagnostico.setStyleSheet("color: green")
			self.label_diagnostico.setText('Benigno')
		else:
			self.label_diagnostico.setStyleSheet("color: red")
			self.label_diagnostico.setText('Sospechoso')



		print('A: ', indice_asimetria)
		print('B: ', indice_borde)
		print('C: ', indice_color)
		print('D: ', indice_diametro)


	def f_ayuda(self):
		msg = QMessageBox()
		msg.setText("Índice de Asimetría (A): Se corresponde con el número de ejes asimétricos que presenta el lunar. Su rango de valores es 0-2, siendo 0 un lunar simétrico, 1 un lunar con asimetría a lo largo de solo uno de sus ejes, y 2 un lunar con asimetría en ambos.\n\nÍndice de Bordes (B): Se corresponde con el número de segmentos del borde que sean considerados como irregulares. Primero se segmenta la imagen del lunar en 8 partes iguales, y luego se analiza cada una por separado. Aquellas consideradas como irregulares suman 1 al valor de B, mientras que las regulares 0. Por ende, el rango de valores para B es 0-8.\n\nÍndice de Color (C): Se corresponde con el número de colores que presenta el lunar. Para todo lunar, se considera que sus posibles colores son: Negro, Blanco, Rojo, Marrón Claro, Marrón Oscuro y Gris Azulado. Entonces, el rango de valores para C es 1-6, siendo 1 un lunar que presenta un único color y 6 un lunar que presenta todos los colores.\n\nÍndice de Diámetro (D): Se corresponde con la medida del diámetro del lunar. Según el valor del diámetro, se calcula un parámetro auxiliar M, y en base a este último se define un rango de 1-5 para el valor de D.\n\nFórmula del TDS: TDS=1.3A+0.1B+0.5C+0.5D\n\nSi el TDS es menor a 5,65, el lunar se clasifica como benigno. De lo contrario, se lo clasifica como sospechoso.\n\n\nDISCLAIMER: HERRAMIENTA PARA OBTENER INFORMACIÓN TEMPRANA. NO ES UNA HERRAMIENTA QUE SE ENCUENTRE VALIDADA O CERTIFICADA, POR LO QUE NO TIENE VALIDEZ MÉDICA. ANTE CUALQUIER DUDA O MAYOR DETALLE SOBRE SU DIAGNÓSTICO, CONSULTE A UN DERMATÓLOGO")
		msg.setIcon(QMessageBox.Information)
		msg.setWindowTitle("Acerca de")
		msg.setStandardButtons(QMessageBox.Ok)
		msg.exec()




app = QApplication(sys.argv)
UIWindow = INICIO()
app.exec()
