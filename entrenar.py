import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout,Flatten, Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K

K.clear_session()

data_entrenamiento = "./data/Entrenamiento"
data_validacion = "./data/Validacion"

epocas = 100                 #iteraciones entrenamiento
altura, longitud = 100,100
batch_size=10              #Cantidad de imagenes * cada paso
pasos=32                  #Cant de pasos por epoca
pasos_validacion=51       
filtroConv1=32
filtroConv2=64 
tamano_filtro1=(3,3)
tamano_filtro2=(2,2)
tamano_pool=(2,2)
clases=2                      #Cantidad de tipos o clases de cosas por diferecniar
lr=0.005

##Pre Procesamiento de imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

validacion_datagen= ImageDataGenerator(
    rescale=1./255
)

imagen_entrenamieto=entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura,longitud),
    batch_size=batch_size,
    class_mode="categorical"
)

imagen_validacion = validacion_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura,longitud),
    batch_size=batch_size,
    class_mode="categorical"
)

#Crear RED CNN

cnn=Sequential()

cnn.add(Convolution2D(filtroConv1,tamano_filtro1,padding="same",input_shape=(altura,longitud,3),activation="relu"))

cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtroConv2,tamano_filtro2,padding="same",activation="relu"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())
cnn.add(Dense(256,activation="relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases,activation="softmax"))

cnn.compile(loss="categorical_crossentropy",optimizer=optimizers.Adam(learning_rate=lr),metrics=["accuracy"])

cnn.fit(imagen_entrenamieto,steps_per_epoch=pasos,epochs=epocas,validation_data=imagen_validacion,validation_steps=pasos_validacion)

dir="./modelo/"

if not os.path.exists(dir):
    os.mkdir(dir)
cnn.save("./modelo/modelo.h5")
cnn.save_weights("./modelo/pesos.h5")