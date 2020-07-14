import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def data_generator(captions_vector, features_dict, num_images_per_batch, vocabulary_size, max_length):
    features, partial_captions, next_word = [], [], []
    num_stored_images = 0
    # loop for ever over images
    while True:
        for key, sequence in captions_vector:
            num_stored_images += 1
            photo = features_dict[key]
            for i in range(1, len(sequence)):
                if sequence[i + 1] == 0:
                    break
                # split into input and output pair
                in_seq, out_seq = sequence[:i], sequence[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocabulary_size)[0]
                # store
                features.append(photo)
                partial_captions.append(in_seq)
                next_word.append(out_seq)

                # yield the batch data
                if num_stored_images == num_images_per_batch:
                    yield [[np.array(features), np.array(partial_captions)], np.array(next_word)]
                    features, partial_captions, next_word = [], [], []
                    num_stored_images = 0
