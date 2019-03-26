# ** IMPORTAÇÕES ** #
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image


# ** REDE NEURAL ** #

classificador = Sequential()

# Convolução #
classificador.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3),
                         activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2, 2)))

classificador.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3),
                         activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2, 2)))

classificador.add(Flatten())

# Rede densa #

classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units=1, activation='sigmoid'))

classificador.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])

# Geradores de imagens
g_treino = ImageDataGenerator(rotation_range=7,
                              horizontal_flip=True,
                              shear_range=0.2,
                              height_shift_range=0.07,
                              zoom_range=0.2,
                              rescale=1./255)

g_teste = ImageDataGenerator(rescale=1./255)

# Carregamento das bases

b_treino = g_treino.flow_from_directory('D:/Projetos/PycharmProjects/\
deeplearning/gato-dog/dataset/training_set',
                                        target_size=(64, 64), batch_size=32,
                                        class_mode='binary')

b_teste = g_teste.flow_from_directory('D:/Projetos/PycharmProjects/\
deeplearning/gato-dog/dataset/test_set',
                                      target_size=(64, 64), batch_size=32,
                                      class_mode='binary')


# classificador.fit_generator(b_treino, steps_per_epoch=400/32,
#                  epochs=5, validation_data=b_teste, validation_steps=100)

img_teste = image.load_img('D:/Projetos/PycharmProjects/deeplearning/\
gato-dog/dataset/test_set/gato/cat.3506.jpg',
                           target_size=(64, 64))

img_teste = image.img_to_array(img_teste)
img_teste /= 255
img_teste = np.expand_dims(img_teste, axis=0)

prev_teste = classificador.predict(img_teste)

resultado_img_teste = "Gato" if prev_teste > 0.5 else "Cachorro"
print("A imagem inserida é de um: {}".format(resultado_img_teste))
