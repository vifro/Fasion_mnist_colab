# Fasion_mnist_colab
Fashion mnist - CNN 


## Get the sample.
First of all, we need to download the data set. The divide it in

    --Train 5/7      ~ 0.714% 
    --Validation 1/7 ~ 0.142%
    --Test 1/7       ~ 0.142%

Validation will be used for validating the development during training epochs. This will be our guide line in how to tune our parameters and to decide which model perform best during training.

The Test sample will be used to evaluate the finished model, since a unseen/unbiased test sample must be used to measure the performance on the model.

Also, a sanity check over the data. Scanning missing values, sizes etc. should be made. Mnist_fashion is known to be a complete set, which is why some parts are skipped. But converting the data to a Panda object will help to inspect it further.


# If runnnig on multiple computers, check if channel is represented first or last.
# keras.backend.set_image_data_format('channels_first') 



    #download mnist data and split into train and test sets
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    type(X_train)

    # Test is already done, 
    random_seed = 2019 # Seed for reproducibility
    train_size = float(5/6)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, 
                                                                    train_size=train_size, 
                                                                    random_state=random_seed)

    # Create dictionary of target classes
    label_dict = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }

## Inspect the sizes and a image to see that we have loaded the data set correctly.
    # Inspect the Shape 
    print("train: " + str(X_train.shape))
    print("validation: " + str(X_validation.shape))
    print("test: " + str(X_test.shape))

    # Number of samples
    print("Number of training samples: " + str(len(y_train)))
    print("Number of validation samples: " + str(len(y_validation)))
    print("Number of test samples: " + str(len(y_test)))


## Normalize

Since the input data consists of image pixels, ranging from 0-255, needs to be normalized. A neural network does not know that pixel values between 0-255 just represent the gray scale and will therefore threat them as different. This will mess up the activation functions so we want to normalize the data on a scale between 0-1.

## Reshape
The model will have an input, and the choice is up to the person building it. The data set contains grey scaled images, which will result in a 1 channel input. (RGB = 3 channels). To make our input work with Keras - Conv2 layer, which is set to work with (batch size, height, width, channels ) or (batch size, channels, height, width) we need to reshape the images to contain the channel value. In our case, 1.

## Convert Labels
Labels are now a number representing 1 out of 10 different articles of clothing. The last layer in the model will be a softmax with 10 neurons, which means that we need to convert the labels to something that the model can understand. This is done by creating a one- hot encoded labels where each answer is a vector with 0s, except for in the place of the correct answe which has the value of 1. This is due to the softmax produces probabilities over the 10 output neurons, and we want to maximize the correct answer to be as close to 1 as possible. This means that all other values in the last layer will ( hopefully) be trained to display 0.

Also, when a one hot encoded label is used, categorical_crossentropy is used. Otherwise sparse_categorical_crossentropy else

    # Reshape data to fit model
    # The channels depend on the backend, 
    # it can be "channel first" or "channel last". 

    nr_channels = 1 # Grey scaled = 1, RGB = 3

    if K.image_data_format() == "channels_first":
        print("Channel first")
        X_train = X_train.reshape(len(X_train),
                                  nr_channels,
                                  X_train[0].shape[0],
                                  X_train[0].shape[1])

        X_validation = X_validation.reshape(len(X_validation),
                                            nr_channels,
                                            X_validation[0].shape[0],
                                            X_validation[0].shape[1])

        X_test = X_test.reshape(len(X_test),
                                nr_channels,
                                X_test[0].shape[0],
                                X_test[0].shape[1])

        input_shape = (nr_channels,
                       X_train[0].shape[0], 
                       X_train[0].shape[1])

    else:
        print("Channel Last")
        X_train = X_train.reshape(len(X_train), 
                                  X_train[0].shape[0],
                                  X_train[0].shape[1],
                                  nr_channels)

        X_validation = X_validation.reshape(len(X_validation), 
                                            X_validation[0].shape[0],
                                            X_validation[0].shape[1], 
                                            nr_channels)

        X_test = X_test.reshape(len(X_test), 
                                X_test[0].shape[0],
                                X_test[0].shape[1], 
                                nr_channels)

        input_shape = (X_train[0].shape[0], 
                       X_train[0].shape[1], 
                       nr_channels)


    # Normalize the data.
    X_train = X_train.astype('float32') / 255
    X_validation = X_validation.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # Inspect the answers
    print("\nShape_old_answers: " + str(y_train.shape))
    print("Example of current label: " + str(y_train[0])) # A number representing a label,

    nr_classes = len(label_dict)
    
    # One hot encode the labels of both train and test
    y_train = np_utils.to_categorical(y_train, nr_classes)
    y_validation = np_utils.to_categorical(y_validation, nr_classes)
    y_test = np_utils.to_categorical(y_test, nr_classes)

    # Inspect Gray scaled image, with one channel (RGB has 3). 
    plt.imshow(X_train[0])

    # https://www.kaggle.com/zalando-research/fashionmnist informatin about the data

# Creating the model
The model is created using keras API with a help function for making multiple models more easy. 

    def buildCNN(input_shape, nr_classes, dropout_rate=0.2, kernel_size=(3, 3), strides=(1,1), 
                 optimizer=Adam, lr=0.001, Bn=False, Zp= False, 
                 activations=['relu', 'relu'], loss=categorical_crossentropy, 
                 metrics=[''], conv_layer=[32, 64], hidden_layer=[128],reg=False):
        """
        Specify how many layers your cnn should contain, and how many hidden layers.
        Add more information if you dont want standard settings.
            Input: 
                dropout_rate:
                    Dropout rate, added before every fully connected.

                kernel_size:
                    The kernel size (window size) for the CNN

                strides:
                    Strides, the step size for the kernel.

                optimizer:
                    Optimizer for compiling, eg. Adam, SGD, Rmstop etc. 

                lr:
                    Initial learning rate for the model

                Bn:
                    If Batch normalization should be used

                Zp:
                    If Zero padding should be used

                activations:
                    List of activation functions, if empty. Relu.

                loss:
                    Loss function, binary, cross_entropy etc.

                metrics:
                    list of strings describing what metric to use

                conv_layer:
                    The size and how many convolutional layers. 

                hidden_layer:
                    Size and number of hidden layers

                nr_classes:
                    Number of output neurons.
            Return:
                A  compiled model with specified input values.

                FIX CALLBACK  FOR EARLY STOPPING SO THAT EVERY FUCKING MODEL CAN BE
                EVALUATED.

        """
        # If L1 and L2 reg should be used
        if reg:
            kernel_reg=regularizers.l1_l2(l1=0.0005, l2=0.0005)
        else:
            kernel_reg=None

        # Set the activations for the different parts of the CNN
        if activations:
            conv_act = activations[0] # Activation for convolutional layers
            hid_act = activations[1] # Activation for hidden layer

        model = Sequential()

        # For each element in the conv_layer, Create one.
        for index, filters in enumerate(conv_layer):
            if Zp:
                model.add(ZeroPadding2D((1,1), input_shape=input_shape))
                model.add(Conv2D(filters, kernel_size=kernel_size, strides=strides,
                                 kernel_regularizer=kernel_reg, 
                                 name="conv"+str(index)))
            else:
                model.add(Conv2D(filters, kernel_size=kernel_size, strides=strides,
                                 input_shape=input_shape, name="conv"+str(index)))
            if Bn:
                   model.add(BatchNormalization())
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2), name="max_pool_"+str(index)))

        # Flatted output, so the fully connected gets the right dimensions. 
        model.add(Flatten())

        # Drop out rate
        model.add(Dropout(dropout_rate))

        # Create n hidden layers.
        for index, elem in enumerate(hidden_layer):
            model.add(Dense(elem, activation=hid_act, name="Dense"+str(elem) 
                      + "_" + str(index)))

        # Add the output
        model.add(Dense(nr_classes, activation=softmax, name="softmax_output"))

        # Compile before returning 
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)

        return model


## First 4 models where created with 2 convolutional layers and one hidden with 128 neurons.

Testing kernel size and strides
Visual Comparison

    kernel_size(3, 3) with stride (1, 1)
    kernel_size(3, 3) with stride (2, 2)
    kernel_size(4, 4) with stride (1, 1)
    kernel_size(4, 4) with stride (2, 2)

# Check difference of kernel sizes and strides.
    opt = Adam(0.001)
    model_3_1 = buildCNN(input_shape, nr_classes, dropout_rate=0.5, kernel_size=(3, 3), strides=(1,1),
                         optimizer=opt, lr=0.001, Bn=False, Zp= False, activations=['relu', 'relu'],
                         loss=categorical_crossentropy, metrics=['categorical_accuracy'],
                         conv_layer=[32, 64], hidden_layer=[128])
    model_3_1.summary()
    model_3_2 = buildCNN(input_shape, nr_classes, dropout_rate=0.5, kernel_size=(3, 3), strides=(2,2),
                         optimizer=opt, lr=0.001, Bn=False, Zp= False, activations=['relu', 'relu'],
                         loss=categorical_crossentropy, metrics=['categorical_accuracy'],
                         conv_layer=[32, 64], hidden_layer=[128])

    model_4_1 = buildCNN(input_shape, nr_classes, dropout_rate=0.5, kernel_size=(4, 4), strides=(1,1),
                         optimizer=opt, lr=0.001, Bn=False, Zp= False, activations=['relu', 'relu'],
                         loss=categorical_crossentropy, metrics=['categorical_accuracy'],
                         conv_layer=[32, 64], hidden_layer=[128])


    model_4_2 = buildCNN(input_shape, nr_classes, dropout_rate=0.5, kernel_size=(4, 4), strides=(2,2),
                         optimizer=opt, lr=0.001, Bn=False, Zp= False, activations=['relu', 'relu'],
                         loss=categorical_crossentropy, metrics=['categorical_accuracy'],
                         conv_layer=[32, 64], hidden_layer=[128])

    BATCH_SIZE = 32
    EPOCHS = 20

# TRAIN


    history1 = model_3_1.fit(X_train, y_train, 
                           epochs = EPOCHS,
                           batch_size=BATCH_SIZE,
                           validation_data = (X_validation, y_validation))

    history2 = model_3_2.fit(X_train, y_train, 
                           epochs = EPOCHS,
                           batch_size=BATCH_SIZE,
                           validation_data = (X_validation, y_validation))

    history3 = model_4_1.fit(X_train, y_train, 
                           epochs = EPOCHS,
                           batch_size=BATCH_SIZE,
                           validation_data = (X_validation, y_validation))

    history4 = model_4_2.fit(X_train, y_train, 
                           epochs = EPOCHS,
                           batch_size=BATCH_SIZE,
                           validation_data = (X_validation, y_validation))



# Print accuracy


    loss1, acc1 = model_3_1.evaluate(X_test, y_test)
    loss2, acc2 = model_3_2.evaluate(X_test, y_test)
    loss3, acc3 = model_4_1.evaluate(X_test, y_test)
    loss4, acc4 = model_4_2.evaluate(X_test, y_test)


    print("Trained model1, accuracy: {:5.2f}%".format(100*acc1))
    print("Trained model2, accuracy: {:5.2f}%".format(100*acc2))
    print("Trained model3, accuracy: {:5.2f}%".format(100*acc3))
    print("Trained model4, accuracy: {:5.2f}%".format(100*acc4))

# OUTPUT
    Trained model1, accuracy: 91.45%
    Trained model2, accuracy: 83.99%
    Trained model3, accuracy: 90.68%
    Trained model4, accuracy: 83.87%

# Check accuracy, loss, classification and overfitting of each.
# Visualize
    classification_and_plot(history1, y_test, model_3_1.predict(X_test), 
                            label_dict, history1.epoch[-1] + 1)
    classification_and_plot(history2, y_test, model_3_2.predict(X_test), 
                            label_dict, history2.epoch[-1] + 1)
    classification_and_plot(history3, y_test, model_4_1.predict(X_test), 
                            label_dict, history3.epoch[-1] + 1)
    classification_and_plot(history4, y_test, model_4_2.predict(X_test), 
                            label_dict, history4.epoch[-1] + 1)
                            

# Conclusion
## Strides:

First off we can easily see that the strides and kernel sizes effect both accuracy and learning rate. The strides of (2, 2) affect the accuracy in a negative way. We have small images, and strides of 2 makes our represenations miss important features. The overall best performer was the kernel size of (3, 3) and strides(2, 2).
Overfitting

From this run we also see that the system is not overfitting to much, which implies that our model could be more complex. Since we have not used any of the techniques to reduce overfitting, improving training accuracy gives us something to work with and will probably also yield in a better result.


# Trying more complex models w/out Batch normalization and ZeroPadding
Lets extend our model to a 3 Layer convolutional with the best performing strides and kernel size.

    model_basic = buildCNN(input_shape, nr_classes, dropout_rate=0.5, kernel_size=(3, 3), 
                           strides=(1,1),optimizer=opt, lr=0.001, Bn=False, 
                           Zp= False, activations=['relu', 'relu'],
                           loss=categorical_crossentropy, 
                           metrics=['categorical_accuracy'],
                           conv_layer=[32, 64, 128], 
                           hidden_layer=[128])

    model_normalization = buildCNN(input_shape, nr_classes, dropout_rate=0.5, kernel_size=(3, 3), 
                                   strides=(1,1),optimizer=opt, lr=0.001, Bn=True, 
                                   Zp= True, activations=['relu', 'relu'],
                                   loss=categorical_crossentropy, 
                                   metrics=['categorical_accuracy'],
                                   conv_layer=[32, 64, 128], 
                                   hidden_layer=[128])

    model_basic.summary()
    model_normalization.summary()

## Training
    EPOCHS = 20
    BATCH_SIZE = 32

    history_basic = model_basic.fit(X_train, y_train, 
                                    epochs = EPOCHS,
                                    batch_size=BATCH_SIZE,
                                    validation_data = (X_validation, y_validation))

    history_normalization = model_normalization.fit(X_train, y_train, 
                                                    epochs = EPOCHS,
                                                    batch_size=BATCH_SIZE,
                                                    validation_data = (X_validation, y_validation))

## Basic model.
    Epoch 1/20
    50000/50000 [==============================] - 15s 308us/step - loss: 0.6648 - categorical_accuracy: 0.7516 - val_loss: 0.4673 - val_categorical_accuracy: 0.8274
    Epoch 2/20
    50000/50000 [==============================] - 15s 294us/step - loss: 0.4893 - categorical_accuracy: 0.8198 - val_loss: 0.4059 - val_categorical_accuracy: 0.8461 
    ....
    Epoch 19/20
    50000/50000 [==============================] - 15s 294us/step - loss: 0.2649 - categorical_accuracy: 0.9018 - val_loss: 0.2962 - val_categorical_accuracy: 0.8919
    Epoch 20/20
    50000/50000 [==============================] - 15s 293us/step - loss: 0.2642 - categorical_accuracy: 0.9013 - val_loss: 0.2980 - val_categorical_accuracy: 0.8900

## Batch normalization
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/20
    50000/50000 [==============================] - 22s 447us/step - loss: 0.4640 - categorical_accuracy: 0.8289 - val_loss: 0.3381 - val_categorical_accuracy: 0.8745
    Epoch 2/20
    50000/50000 [==============================] - 21s 428us/step - loss: 0.3202 - categorical_accuracy: 0.8817 - val_loss: 0.3040 - val_categorical_accuracy: 0.8854
    ...
    Epoch 19/20
    50000/50000 [==============================] - 21s 412us/step - loss: 0.1319 - categorical_accuracy: 0.9493 - val_loss: 0.2243 - val_categorical_accuracy: 0.9224
    Epoch 20/20
    50000/50000 [==============================] - 20s 410us/step - loss: 0.1260 - categorical_accuracy: 0.9531 - val_loss: 0.2260 - val_categorical_accuracy: 0.9270
    
As we can see, batch normalization imporved our overfitting problem

# Fine tuning parameters
There are multiple ways to fine tune hyper parameters. Lets see if the learning rate is to big for the model to get even better results. So we run the same process again, with some more epochs.. But with a learning rate decay.


    # Reduce the learning rate and run for a few epochs.
    EPOCHS = 30
    BATCH_SIZE = 32 # increase for more generalized learning and decreased accuracy

    learning_rate = 0.001
    decay_rate = learning_rate / EPOCHS
    opt = Adam(lr=learning_rate, epsilon=None, decay=decay_rate)

    model_normalization_decay = buildCNN(input_shape, nr_classes, dropout_rate=0.5, 
                                         kernel_size=(3, 3), strides=(1,1), 
                                         optimizer=opt, lr=0.001, Bn=True, Zp=True, 
                                         activations=['relu', 'relu'],
                                         loss=categorical_crossentropy, 
                                         metrics=['categorical_accuracy'],
                                         conv_layer=[32, 64, 128], 
                                         hidden_layer=[128])


    history_normalization_decay = model_normalization_decay.fit(X_train, y_train, 
                                                    epochs = EPOCHS,
                                                    batch_size=BATCH_SIZE,
                                                    validation_data = (X_validation, y_validation))

## Training
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/30
    50000/50000 [==============================] - 23s 457us/step - loss: 0.4898 - categorical_accuracy: 0.8215 - val_loss: 0.3201 - val_categorical_accuracy: 0.8841
    Epoch 2/30
    50000/50000 [==============================] - 21s 429us/step - loss: 0.3291 - categorical_accuracy: 0.8782 - val_loss: 0.3002 - val_categorical_accuracy: 0.8850
    ...
    Epoch 29/30
    50000/50000 [==============================] - 21s 417us/step - loss: 0.0657 - categorical_accuracy: 0.9758 - val_loss: 0.2274 - val_categorical_accuracy: 0.9353
    Epoch 30/30
    50000/50000 [==============================] - 21s 418us/step - loss: 0.0623 - categorical_accuracy: 0.9765 - val_loss: 0.2217 - val_categorical_accuracy: 0.9364
## Testing
    Trained final model, accuracy: 92.61%

# Conclusion
By inspecting the values of recall, precision and f1. We can clearly see that the model has no trouble at all classifying trousers but only 80% on shirts.
Reaching Final model

    Step1: Trying a basic model with different strides and kernel sizes ( Should have tried more for being correct)
    Step2: Trying to build a more complex model to overfit the data
    Step3: Trying to Normalize to reduce the overfitting
    
## What to do next.
Try L1 and L2 regularization on each layer and see if we reduce the overfitting of the validation data.
# Reduce the learning rate and run for a few epochs.
    EPOCHS = 30
    BATCH_SIZE = 32 # increase for more generalized learning and decreased accuracy

    learning_rate = 0.001
    decay_rate = learning_rate / EPOCHS
    opt = Adam(lr=learning_rate, epsilon=None, decay=decay_rate)

    model_normalization_decay_l1_l2 = buildCNN(input_shape, nr_classes, dropout_rate=0.5, 
                                               kernel_size=(3, 3), strides=(1,1), 
                                               optimizer=opt, lr=0.001, Bn=True, Zp=True, 
                                               activations=['relu', 'relu'],
                                               loss=categorical_crossentropy, 
                                               metrics=['categorical_accuracy'],
                                               conv_layer=[32, 64, 128], 
                                               hidden_layer=[128], reg=True)


    history_normalization_decay_l1_l2 = model_normalization_decay_l1_l2.fit(X_train, 
                                                                            y_train, 
                                                                            epochs = EPOCHS,
                                                                            batch_size=BATCH_SIZE,
                                                                            validation_data = (X_validation, y_validation))

## Final score:
    Test data: 92.21%
# Conclusion
This is not a perfect model. A lot of things could have been improved and tried. As seen, L1 and L2 should be tried with different settings since it seems to regularize, but not enough.

    Learning rates
    Different optimizers
    Different Dropout
    Different activation functions
    etc
    Train for more epochs with a callback function for ealy stopping.
    Add one more layers and prevent overfitting on that.
