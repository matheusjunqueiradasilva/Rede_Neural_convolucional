
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator 

classifier = Sequential()

classifier.add(Conv2D(32,(3,3), input_shape=(64,64,3),activation = 'relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(units=128,activation='relu'))

classifier.add(Dense(units=1,activation='sigmoid'))

classifier.compile(optimizer = 'adam', loss ='binary_crossentropy',metrics=['accuracy'])

gen_treino_img = ImageDataGenerator(rescale=1./255,shear_range=0.2,
                                        zoom_range=0.2,horizontal_flip=True)

gen_teste_img = ImageDataGenerator(rescale=1./255)

treino_set = gen_treino_img.flow_from_directory('training_set',target_size = (64,64),
                                       batch_size=32, class_mode='binary')


test_set = gen_teste_img.flow_from_directory('test_set',target_size = (64,64),
                                    batch_size=32,class_mode='binary')


classifier.fit_generator(treino_set,steps_per_epoch=8000, epochs= 20,
                         validation_data= test_set, validation_steps=2000)


















