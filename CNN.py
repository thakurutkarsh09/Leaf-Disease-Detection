#convolutional layer-1
detection.add(Conv2D(64,(3,3),padding='same',input_shape=(48,48,3)))
detection.add(BatchNormalization())
detection.add(Activation('relu'))
detection.add(MaxPooling2D(pool_size=(2,2)))
detection.add(Dropout(0.25))

#2 -convolutional layer-2
detection.add(Conv2D(128,(5,5),padding='same'))
detection.add(BatchNormalization())
detection.add(Activation('relu'))
detection.add(MaxPooling2D(pool_size=(2,2)))
detection.add(Dropout(0.25))

#3 -convolutional layer-3
detection.add(Conv2D(256,(3,3),padding='same'))
detection.add(BatchNormalization())
detection.add(Activation('relu'))
detection.add(MaxPooling2D(pool_size=(2,2)))
detection.add(Dropout(0.25))

#4 -convolutional layer-4
detection.add(Conv2D(512,(3,3),padding='same'))
detection.add(BatchNormalization())
detection.add(Activation('relu'))
detection.add(MaxPooling2D(pool_size=(2,2)))
detection.add(Dropout(0.25))

#5 -convolutional layer-5
detection.add(Conv2D(512,(3,3),padding='same'))
detection.add(BatchNormalization())
detection.add(Activation('relu'))
detection.add(MaxPooling2D(pool_size=(2,2)))
detection.add(Dropout(0.25))

detection.add(Flatten())
detection.add(Dense(256))
detection.add(BatchNormalization())
detection.add(Activation('relu'))
detection.add(Dropout(0.25))

detection.add(Dense(512))
detection.add(BatchNormalization())
detection.add(Activation('relu'))
detection.add(Dropout(0.25))

detection.add(Dense(15,activation='softmax'))
optimum=Adam(lr=0.005)
#lr-learning rate
detection.compile(optimizer=optimum,loss='categorical_crossentropy',metrics=['accuracy'])
