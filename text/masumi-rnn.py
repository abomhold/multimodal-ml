import numpy as np
import tensorflow as tf
from os import access
# TODO:
"""
Currently this model grabs a seq of text and uses the next word as the label
Need to redo this and relate to gender
"""

def create_dictionaries(unique_chars):
    return {char: i for i, char in enumerate(unique_chars)}, {i: char for i, char in enumerate(unique_chars)}


def one_hot_vector(sequences, labels, num_classes):
    # First convert the sequences & labels into numpy arrays
    X = np.array(sequences)
    y = np.array(labels)

    # Second convert the sequence & labels into one-hot vector representation
    X = tf.keras.utils.to_categorical(X, num_classes=num_classes)
    y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

    return X, y


def create_sequence_and_label(raw_text, char_to_index, seq_length):
    sequences = []
    labels = []

    for i in range(len(raw_text) - seq_length):
        # Select the sequence of characters
        sequence = raw_text[i:i + seq_length]
        # Select the next output character
        label = raw_text[i + seq_length]
        # Store the sequence & labels
        sequences.append([char_to_index[char] for char in sequence])
        labels.append(char_to_index[label])

    return sequences, labels


def simple_RNN(X, unique_chars):
    model = tf.keras.Sequential()
    # units=50 means how many neurons I'm using
    model.add(tf.keras.layers.SimpleRNN(units=75, input_shape=(X.shape[1], X.shape[2]), activation='relu'))
    model.add(tf.keras.layers.Dense(units=len(unique_chars), activation='softmax'))

    '''
    Compile the mode for training using the following parameters
    Optimization: Adam optimizer (used to minimize the loss function during the training process)
    Loss: Categorical Cross entropy
    Metrics: Accuracy
    '''
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # summarize defined model
    # model.summary()
    tf.keras.utils.plot_model(model=model, to_file='model_RNN.png', show_shapes=True)

    return model


def lstm_and_softmax(X, unique_chars):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=75, input_shape=(X.shape[1], X.shape[2])))
    model.add(tf.keras.layers.Dense(len(unique_chars), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()
    tf.keras.utils.plot_model(model=model, to_file='model_LSTM.png', show_shapes=True)
    return model


def generate_text(model, vocab_length, seq_length, seed_text, n_chars, char_to_index, index_to_char):
    output_text = seed_text
    for i in range(n_chars):
        # Map each character of the seed text to its index position
        x = np.array([[char_to_index[char] for char in output_text[-seq_length:]]])  # negative means start from the tail of list
        x_one_hot = tf.keras.utils.to_categorical(x, num_classes=vocab_length)
        # Use the training model to make prediction
        prediction = model.predict(x_one_hot, verbose=0)
        next_index = np.argmax(prediction)
        next_char = index_to_char[next_index]
        output_text += next_char

    return output_text


def main():
    with open('/home/text/rhyme.text', 'r') as file:
        raw_text = file.read()
    seq_length = 10
    # Step 1: Create vocabulary
    print("Fetching the unique characters from the data...")
    unique_chars = sorted(list(set(raw_text)))
    print("Creating char_to_index and index_to_char...")
    char_to_index, index_to_char = create_dictionaries(unique_chars)

    # Step 2: Create input sequences and corresponding labels
    print("Creating sequences and labels")
    sequences, labels = create_sequence_and_label(raw_text, char_to_index, seq_length)

    # Step 3: Encode text to on-hot vector representation
    print("Vectorizing our sequence and labels")
    X, y = one_hot_vector(sequences, labels, len(unique_chars))
    # X.shape is (409, 3, 37) --> (num of seq, length of input seq, length of vocab)

    # Step 4: Define a RNN model for character-base text generation
    # Model 1: Simple RNN w/ ReLU activation and soft max
    # Model 2: LSTM w/ soft max

    rnn = simple_RNN(X, unique_chars)
    lstm = lstm_and_softmax(X, unique_chars)

    # Step 5: Train the model using input sequences (X) and corresponding labels (y) for 100 epochs
    print("Training both RNN and LSTM model...")
    rnn.fit(X, y, epochs=100, verbose=0)
    lstm.fit(X, y, epochs=100, verbose=0)

    # Step 6: Model prediction
    start_seq = 'king was i'

    # Generate text using both RNN & LSTM
    print("Generating text using both trained models...")
    generated_text_RNN = generate_text(rnn, len(unique_chars), seq_length, start_seq, 150, char_to_index, index_to_char)
    generated_text_LSTM = generate_text(lstm, len(unique_chars), seq_length, start_seq, 150, char_to_index, index_to_char)

    print(f'RNN Generated Text: {generated_text_RNN}')
    print(f'LSTM Generated Text: {generated_text_LSTM}')


if __name__ == '__main__':
    main()