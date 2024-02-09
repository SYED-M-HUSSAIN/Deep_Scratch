from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Create a Sequential model
model = Sequential()

# Add a 1D convolutional layer
model.add(Conv1D(64, 3, activation='relu', input_shape=(100, 1)))

# Add a max pooling layer
model.add(MaxPooling1D(2))

# Add another 1D convolutional layer
model.add(Conv1D(128, 3, activation='relu'))

# Add a max pooling layer
model.add(MaxPooling1D(2))

# Flatten the feature maps
model.add(Flatten())

# Add fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))  # Assuming 10 output classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()
