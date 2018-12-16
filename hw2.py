import sys, os, datetime
import numpy as np
import argparse

import keras
from keras import models, layers, backend
from keras.layers import Dense, Activation, Dropout, Flatten,\
    Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# confirm Keras sees the GPU
assert len(backend.tensorflow_backend._get_available_gpus()) > 0

models_dir = 'models'

#https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720

def AndreaNet(input_shape, num_classes):
    
    print('\AndreaNet v1 model')
    model = models.Sequential()
    
    model.add(Conv2D(6, kernel_size=(11, 11), strides=(2, 2), activation='tanh', input_shape=input_shape, padding="same"))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(16, kernel_size=(8, 8), strides=(2, 2), activation='tanh', padding='valid'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(120, kernel_size=(8, 8), strides=(2, 2), activation='tanh', padding='valid'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(240, kernel_size=(2, 2), strides=(1, 1), activation='tanh', padding='valid'))
    model.add(BatchNormalization())
    
    model.add(Flatten())

    model.add(Dense(2048, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(512, activation='tanh'))
    model.add(BatchNormalization())

    model.add(Dense(num_classes, activation='softmax'))

    optimizer = 'adam' #alternative 'SGD'
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    
    return model

def AndreaNet2(input_shape, num_classes):
    
    print('AndreaNet v2 model')
    model = models.Sequential()
    
    model.add(Conv2D(6, kernel_size=(11, 11), strides=(2, 2), activation='relu', input_shape=input_shape, padding="same"))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(16, kernel_size=(8, 8), strides=(2, 2), activation='relu', padding='valid'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(120, kernel_size=(8, 8), strides=(2, 2), activation='relu', padding='valid'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(240, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='valid'))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    
    model.add(Dense(2048, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    model.add(Dense(512, activation='tanh'))
    model.add(BatchNormalization())
    
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = 'SGD' #alternative 'SGD'
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    
    return model
 
def AndreaNet3(input_shape, num_classes):
    
    print('\nAndreaNet v3 model')
    model = models.Sequential()
    
    model.add(Conv2D(12, kernel_size=(11, 11), strides=(2, 2), activation='relu', input_shape=input_shape, padding="same"))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(32, kernel_size=(8, 8), strides=(2, 2), activation='relu', padding='valid'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(240, kernel_size=(8, 8), strides=(2, 2), activation='relu', padding='valid'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(360, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='valid'))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(num_classes, activation='softmax'))

    #optimizer = 'SGD' #alternative 'SGD'
    optimizer = "adam"
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    
    return model


def LeNet(input_shape, num_classes):
    
    print('\nLeNet model')
    model = models.Sequential()
    
    print('\tC1: Convolutional 6 kernels 5x5')
    model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_shape, padding="same"))
    print('\tS2: Average Pooling 2x2 stride 2x2')
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    print('\tC3: Convolutional 16 kernels 5x5')
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    print('\tS4: Average Pooling 2x2 stride 2x2')
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    print('\tC5: Convolutional 120 kernels 5x5')
    model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    model.add(Flatten())
    print('\tF6: Fully connected, 84 units')
    model.add(Dense(84, activation='tanh'))
    print('\tF7: Fully connected, 10 units')
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = 'adam' #alternative 'SGD'
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    
    return model
        
def AlexNet(input_shape, num_classes):
    # Some details in https://www.learnopencv.com/understanding-alexnet/

    model = models.Sequential()

    # C1 Convolutional Layer 
    model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11),\
     strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # C2 Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # C3 Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # C4 Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # C5 Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Flatten
    model.add(Flatten())

    # D1 Dense Layer
    model.add(Dense(4096, input_shape=(224*224*3,)))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # D2 Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # D3 Dense Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
 


def savemodel(model,problem):
    if problem.endswith('.h5'):
        filename = problem
    else:
        filename = os.path.join(models_dir, '%s.h5' %problem)
    model.save(filename)
    #W = model.get_weights()
    #print(W)
    #np.savez(filename, weights = W)
    print("\nModel saved successfully on file %s\n" %filename)

    
def loadmodel(problem):
    if problem.endswith('.h5'):
        filename = problem
    else:
        filename = os.path.join(models_dir, '%s.h5' %problem)
    try:
        model = models.load_model(filename)
        print("\nModel loaded successfully from file %s\n" %filename)
    except OSError:    
        print("\nModel file %s not found!!!\n" %filename)
        model = None
    return model

    
### main ###
if __name__ == "__main__":

    nets = {
        "alexnet": AlexNet,
        "lenet": LeNet,
        "andreanet": AndreaNet,
        "andreanet2": AndreaNet2,
        "andreanet3": AndreaNet3,
    }
    
    parser = argparse.ArgumentParser(description='HW2')
    parser.add_argument('-seed', type=int, help='random seed', default=0)
    parser.add_argument('-draw', type=int, help='draw the model', default=0)
    parser.add_argument('-net', type=str, help='net type %s' % str(nets.keys()), required=True)
    parser.add_argument('-W', type=int, help='target size width', required=True)
    parser.add_argument('-H', type=int, help='target size heigth', required=True)
    args = parser.parse_args()
    
    if args.net.lower() not in nets.keys():
        print("invalid net %s" % args.net)
        exit(1)
    
    problem = "%s_argos_%dx%d" % (args.net.lower(), args.W, args.H)
    
    print("Creating train generator")
    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(
        directory="./sc5/",
        target_size=(args.W, args.H),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    
    print("Creating test generator")
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
        directory="./sc5_test/",
        target_size=(args.W, args.H),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    
    num_classes = 24
    input_shape = (args.W, args.H, 3)
    print (input_shape)
    # Load or create model
    model = loadmodel(problem)
    if model==None:
        model = nets[args.net.lower()](input_shape, num_classes)
    model.summary()
    
    if args.draw != 0:
        from keras.utils import plot_model
        plot_model(model, to_file='%s.png' % problem)
        exit()
    
    # Set random seed
    if args.seed==0:
        dt = datetime.datetime.now()
        rs = int(dt.strftime("%y%m%d%H%M"))
    else:
        rs = args.seed
    np.random.seed(rs)
    
    print("\nRandom seed %d" %rs)
    
    print("\nTraining ...")
    
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = test_generator.n // test_generator.batch_size

     # Train
    try:
       model.fit_generator(generator=train_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            epochs=100
        )
    except KeyboardInterrupt:
        pass

    print("\n\nEvaluation ...")

    try:
        score = model.evaluate_generator(generator=test_generator, steps=STEP_SIZE_VALID)
        print("Test loss: %f" %score[0])
        print("Test accuracy: %f" %score[1])
    except KeyboardInterrupt:
        pass

    # Save the model
    savemodel(model,problem)

