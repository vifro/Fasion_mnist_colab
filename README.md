# Fasion_mnist_colab
Fashion mnist - CNN 


First 4 models where created.

Testing kernel size and strides
Visual Comparison

    kernel_size(3, 3) with stride (1, 1)
    kernel_size(3, 3) with stride (2, 2)
    kernel_size(4, 4) with stride (1, 1)
    kernel_size(4, 4) with stride (2, 2)

In [10]:

opt = Adam(0.001)
# Check difference of kernel sizes and strides.
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



Trained model1, accuracy: 91.45%
Trained model2, accuracy: 83.99%
Trained model3, accuracy: 90.68%
Trained model4, accuracy: 83.87%
