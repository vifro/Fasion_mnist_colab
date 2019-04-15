# Fasion_mnist_colab
Fashion mnist - CNN 


First 4 models where created.

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
                            

## kernel_size(3, 3) with stride (1, 1)



    [INFO] evaluating network...
                  precision    recall  f1-score   support

     T-shirt/top       0.88      0.85      0.87      1000
         Trouser       1.00      0.98      0.99      1000
        Pullover       0.87      0.86      0.86      1000
           Dress       0.93      0.92      0.92      1000
            Coat       0.86      0.84      0.85      1000
          Sandal       0.98      0.98      0.98      1000
           Shirt       0.72      0.80      0.76      1000
         Sneaker       0.94      0.98      0.96      1000
             Bag       0.99      0.98      0.98      1000
      Ankle boot       0.98      0.95      0.97      1000

       micro avg       0.91      0.91      0.91     10000
       macro avg       0.92      0.91      0.92     10000
    weighted avg       0.92      0.91      0.92     10000

    <Figure size 432x288 with 0 Axes>


