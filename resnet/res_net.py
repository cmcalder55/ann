
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, ZeroPadding2D, Add
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.engine.input_layer import Input
from keras.initializers import glorot_uniform

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",  "dog", "frog", "horse", "ship", "truck"]
N_CLASSES = len(CLASSES)
BATCH_SIZE = N_FILTERS = 64

class ResNet50():

    def __init__(self, x_test, y_test, x_train, y_train):

        self.model = self.create_resnet_model()
        self.X_test, self.Y_test, self.X_train, self.Y_train = self.init_data(x_test, y_test, x_train, y_train)
        self.steps_per_epoch = self.Y_train.shape[0]//BATCH_SIZE
        self.validation_steps = self.Y_test.shape[0]//BATCH_SIZE

    def show_random_sample(self):
        # show random images from training data
        rows, cols = 2, 8
        fig = plt.figure(figsize=(2*cols - 1, 2.5*rows - 1))
        for i in range(cols):
            for j in range(rows):
                random_index = np.random.randint(0, len(y_train))
                ax = fig.add_subplot(rows, cols, i*rows + j + 1)
                ax.grid('off')
                ax.axis('off')
                ax.imshow(x_train[random_index, :])
                ax.set_title(CLASSES[y_train[random_index, 0]])
                plt.show()    

    def init_data(self, x_test, y_test, x_train, y_train):

        print("Train samples:", x_train.shape, y_train.shape)
        print("Test samples:", x_test.shape, y_test.shape)
        
        X_train = x_train.astype('float32')/255
        X_test = x_test.astype('float32')/255

        Y_train = np_utils.to_categorical(y_train, N_CLASSES)
        Y_test = np_utils.to_categorical(y_test, N_CLASSES)
        
        return X_test, Y_test, X_train, Y_train

    def create_generators(self):
        
        x_val, partial_x_train = self.X_train[:10000], self.X_train[10000:]
        y_val, partial_y_train = self.Y_train[:10000], self.Y_train[10000:]
        
        gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3, height_shift_range=0.08, zoom_range=0.08)
        train_generator = gen.flow(partial_x_train, partial_y_train, batch_size=BATCH_SIZE)
        
        val_gen = ImageDataGenerator()    
        val_generator = val_gen.flow(x_val, y_val, batch_size=BATCH_SIZE)

        return train_generator, val_generator

    def create_resnet_model(self):

        # Define the input as a tensor with shape input_shape
        X_input = Input((32,32,3))

        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)

        # ---------------Stage 1------------------------
        X = Conv2D(N_FILTERS, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # ----------Stage 2 for ResNet-------------------

        # Save the input value. You'll need this later to add back to the main path.
        X_shortcut = X

        # conv2d layer --> batch norm layer --> relu activation layer
        X = Conv2D(N_FILTERS, (3, 3), strides = (1, 1), name = 'conv2', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = 'bn_conv2')(X)
        X = Activation('relu')(X)
        # add extra conv2d layer chain to get equal size when combining Stage 1 and Stage 2
        X = Conv2D(N_FILTERS, (3, 3), strides = (1, 1), name = 'conv3', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = 'bn_conv3')(X)
        X = Activation('relu')(X)
        # final conv2d layer and normalization
        X = Conv2D(N_FILTERS, (3, 3), strides = (1, 1), name = 'conv4', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = 'bn_conv4')(X)
        #Add X_shortcut before doing final activation
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        # output layer
        X = Flatten()(X)
        X = Dense(N_FILTERS, activation='softmax', name=f'fc{N_CLASSES}', kernel_initializer = glorot_uniform(seed=0))(X)

        # Create model
        print()
        model = Model(inputs = X_input, outputs = X, name='ResNet50')
        model.summary()
        print()

        return model 

    def _fit(self):

        train_generator, val_generator = self.create_generators()

        self.model.compile(loss='categorical_crossentropy', 
                            optimizer=Adam(), 
                            metrics=['accuracy']
                            )
        self.model.fit(train_generator, 
                epochs=10, 
                validation_data=val_generator,
                # steps_per_epoch=steps_per_epoch, 
                # validation_steps=validation_steps
                )

    def _evaluate(self):
        score = self.model.evaluate(self.X_test, self.Y_test)
        print()
        print('Test accuracy: ', score[1])
        print()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
nn = ResNet50(x_test, y_test, x_train, y_train)

# nn.show_random_sample()

nn._fit()
nn._evaluate()