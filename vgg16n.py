# vgg16 modified to Keras2

import numpy as np

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Lambda, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras.optimizers import Adam


vgg_mean = np.array([123.68, 116.779, 103.939], dtype = np.float32).reshape((3,1,1))

def veg_preprocess(x):
	x = x-vgg_mean
	return x[:, ::-1]
	
class Vgg16n():
	def __init__(self):
		self.FILE_PATH="http://files.fast.ai/models"
		self.create()
		self.get_classes()
	
	def get_classes(self):
		fname = 'imagenet_class_index.json'
		fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir = 'models')
		with open(fpath) as f:
			class_dict=json.load(f)
		self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))] 	
		
	def ConvBlock(self, layers, filters):
		model = self.model
		for i in range(layers):
			model.add(ZeroPadding2D((1,1)))
			model.add(Conv2D(filters, (3,3), data_format='channels_first', activation='relu'))
		model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_first'))
		
	def FCBlock(self):
		model = self.model
		model.add(Dense(4096, activation='relu'))
		model.add(Dropout(0.5))
	
	def create(self):
		model = self.model = Sequential()
		model.add(Lambda(veg_preprocess, input_shape=(3,224,224), output_shape=(3,224,224)))
		print("layers in model: ", len(model.layers))
		print("output shape of layer 0: ", model.layers[0].output.shape)
		self.ConvBlock(2, 64)
		print("output shape of layer 1: ", model.layers[1].output.shape)
		self.ConvBlock(2, 128)
		self.ConvBlock(3, 256)
		self.ConvBlock(3, 512)
		self.ConvBlock(3, 512)
		model.add(Flatten(data_format='channels_first'))
		self.FCBlock()
		self.FCBlock()
		model.add(Dense(1000, activation='softmax'))
		fname = 'vgg16.h5'
		model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))
		
	
	def get_generator(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4, class_mode='categorical'):
		print("in generator")
		return gen.flow_from_directory(path, target_size=(224,224), class_mode=class_mode, batch_size=batch_size, shuffle=shuffle)
		
	
	def compile(self, lr=0.010):
		self.model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
		print("compile over")
		
	def fitvaldata(self, train_generator, val_generator, num_epochs=1, batch_size=4):
		self.model.fit_generator(train_generator, steps_per_epoch=len(train_generator)/batch_size, epochs=num_epochs, validation_data=val_generator, validation_steps=len(val_generator)/batch_size)
	
	def lastlayertuning(self, number_classes):
		model = self.model
		model.pop()
		for layer in model.layers:
			layers.trainable = False
		model.add(Dense(number_classes, activation='softmax'))
		self.compile()
	
	def finetune(self, generator):
		self.lastlayertuning(generator.num_classes)
		classes = list(iter(generator.class_indices))
		
		for c in generator.class_indices:
			classes[generator.class_indices[c]] = c
			
		self.classes = classes
		print("fine tune done")
		
		
		
		
		
		
		
		
		
		
		
		
		