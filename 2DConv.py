from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# Create a Sequential model
model = Sequential()
model.add(Conv2D(2,(3,3),input_shape=(12,12,3)))
# Add max pooling layers
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Conv2D(4,(3,3),activation = 'relu'))
# Add max pooling layers
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(2,(3,3),activation = 'relu'))
# model.add(Conv2D(6,(3,3),activation = 'relu'))
model.add(Flatten())

# Add fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))  # Assuming 10 output classes
# Display the model summary
model.summary()
