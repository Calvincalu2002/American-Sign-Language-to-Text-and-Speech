import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPool1D, BatchNormalization, Activation, Add, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import keras
from keras.saving import register_keras_serializable




data_dict = pickle.load(open(r'C:\Users\anton\Desktop\tensorflow\deploy\orginal\sign_language (2)\sign_language/data.pickle', 'rb'))
data_shapes = [np.shape(item) for item in data_dict['data']]
print(f"Unique shapes: {set(data_shapes)}")

# Ensure uniform shape or pad
try:
    data = np.asarray(data_dict['data'], dtype='float32')  # Try directly converting
except ValueError:
    print("Padding sequences to uniform length...")
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    data = pad_sequences(data_dict['data'], padding='post', dtype='float32')


labels = np.asarray(data_dict['labels'], dtype='int')
num_classes = len(set(labels))
labels = to_categorical(labels, num_classes=num_classes)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

x_train = x_train.reshape((-1, x_train.shape[1], 1))
x_test = x_test.reshape((-1, x_test.shape[1], 1))






# Define the model
@register_keras_serializable(package='CustomModel')
class identity(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, name=None, **kwargs):
        super(identity, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
        self.conv_1 = Conv1D(filters, kernel_size, padding="same", name=f"{name}_conv1" if name else None)
        self.bn_1 = BatchNormalization(name=f"{name}_bn1" if name else None)
        self.act = Activation("relu", name=f"{name}_act" if name else None)
        
        self.conv_2 = Conv1D(filters, kernel_size, padding="same", name=f"{name}_conv2" if name else None)
        self.bn_2 = BatchNormalization(name=f"{name}_bn2" if name else None)
        
        self.add = Add(name=f"{name}_add" if name else None)
        
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.act(x)
        
        x = self.conv_2(x)
        x = self.bn_2(x)
        
        if x.shape[-1] != inputs.shape[-1]:
            inputs = Conv1D(filters=x.shape[-1], kernel_size=1, padding="same")(inputs)
        
        x = self.add([x, inputs])
        x = self.act(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
        })
        return config

@register_keras_serializable(package='CustomModel')
class residual_model(tf.keras.Model):
    def __init__(self, classes, name=None, **kwargs):
        super(residual_model, self).__init__(name=name, **kwargs)
        self.classes = classes
        
        self.conv = Conv1D(64, 7, padding="same")
        self.bn = BatchNormalization()
        self.act = Activation("relu")
        self.pool = MaxPool1D(3)
        
        self.id_1 = identity(64, 3, name="id1")
        self.id_2 = identity(64, 3, name="id2")
        
        self.gape = GlobalAveragePooling1D()
        self.classifier = Dense(classes, activation='softmax')
    
    def call(self, input_val):
        x = self.conv(input_val)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        
        x = self.id_1(x)
        x = self.id_2(x)
        
        x = self.gape(x)
        
        return self.classifier(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "classes": self.classes
        })
        return config
    




model = residual_model(num_classes)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")







# # model = Sequential([
# #     Flatten(input_shape=x_train.shape[1:]),
# #     Dense(128, activation='relu'),
# #     Dropout(0.3),
# #     Dense(64, activation='relu'),
# #     Dropout(0.3),
# #     Dense(num_classes, activation='softmax')
# # ])

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# # history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), batch_size=32)

# # Evaluate the model
# loss, accuracy = model.evaluate(x_test, y_test)
# print(f"Test Accuracy: {accuracy * 100:.2f}%")

# # Save the model
model.save('model.keras')

print("Model saved as 'model.keras'")