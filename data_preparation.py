import os
import pickle

from pathlib import Path

from music21 import corpus, note, chord

import numpy as np

from keras.utils import np_utils


def prepare_data(section, run_id, music_name, seq_len):
    list_of_files, bach_parser = get_bach_chorales()
    print('Amount of files:', len(list_of_files))

    store_folder, run_folder = create_dir_tree(section=section,
                                               run_id=run_id,
                                               music_name=music_name)

    notes, durations = extract_notes(file_list=list_of_files,
                                     parser=bach_parser,
                                     seq_len=seq_len,
                                     store_folder=store_folder)

    n_notes, n_durations, distincts, lookups = save_notes_and_durations(notes, durations, store_folder)

    network_input, network_output = prepare_sequences(notes=notes,
                                                      durations=durations,
                                                      lookups=lookups,
                                                      distincts=distincts,
                                                      seq_len=seq_len)

    return notes, n_notes, durations, n_durations, lookups, distincts, network_input, network_output, run_folder


def save_notes_and_durations(notes, durations, store_folder):
    note_names, n_notes = get_distinct(notes)
    duration_names, n_durations = get_distinct(durations)
    distincts = [note_names, n_notes, duration_names, n_durations]

    with open(os.path.join(store_folder, 'distincts'), 'wb') as distincts_f:
        pickle.dump(distincts, distincts_f)

    note_to_int, int_to_note = create_lookups(note_names)
    durations_to_int, int_to_durations = create_lookups(duration_names)
    lookups = [note_to_int, int_to_note, durations_to_int, int_to_durations]

    with open(os.path.join(store_folder, 'lookups'), 'wb') as lookups_f:
        pickle.dump(lookups, lookups_f)

    return n_notes, n_durations, distincts, lookups


def get_bach_chorales():
    """
    Read Bach's chorales
    :return file_list, parser: list of midi files and parser to read them
    """
    file_list = ['bwv' + str(x['bwv']) for x in corpus.chorales.ChoraleList().byBWV.values()]
    parser = corpus

    return file_list, parser


def extract_notes(file_list, parser, seq_len, store_folder, mode='build'):
    """
    Extract notes names and durations from score
    :param file_list:
    :param parser:
    :param seq_len:
    :param store_folder:
    :param mode:
    :return notes, durations: list of notes names and list of durations
    """
    if mode == 'build':
        notes = []
        durations = []

        for i, file in enumerate(file_list):
            print(i + 1, "Parsing %s" % file)
            original_score = parser.parse(file).chordify()

            notes.extend(['START'] * seq_len)
            durations.extend([0] * seq_len)

            for element in original_score.flat:
                if isinstance(element, note.Note):
                    if element.isRest:
                        notes.append(str(element.name))
                    else:
                        notes.append(str(element.nameWithOctave))

                    durations.append(element.duration.quarterLength)
                if isinstance(element, chord.Chord):
                    notes.append('.'.join(n.nameWithOctave for n in element.pitches))
                    durations.append(element.duration.quarterLength)

        # save notes and durations
        with open(os.path.join(store_folder, 'notes'), 'wb') as notes_file:
            pickle.dump(notes, notes_file)
        with open(os.path.join(store_folder, 'durations'), 'wb') as durations_file:
            pickle.dump(durations, durations_file)
    else:
        with open(os.path.join(store_folder, 'notes'), 'rb') as notes_file:
            notes = pickle.load(notes_file)
        with open(os.path.join(store_folder, 'durations'), 'rb') as durations_file:
            durations = pickle.load(durations_file)

    return notes, durations


def create_dir_tree(section, run_id, music_name):
    """
    Function for creating directory tree to work with dataset
    :param section:
    :param run_id:
    :param music_name:
    :return store_folder: where to store dataset in an appropriate state
    """
    run_folder = 'run/{}/'.format(section)
    run_folder += '_'.join([run_id, music_name])
    store_folder = os.path.join(run_folder, 'store')

    Path(store_folder).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(run_folder, 'output')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(run_folder, 'weights')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(run_folder, 'viz')).mkdir(parents=True, exist_ok=True)

    return store_folder, run_folder


def get_distinct(elements):
    """
    Get all pitch names
    :param elements:
    :return elements_names, n_elements: names of elements and their amount
    """
    element_names = sorted(set(elements))
    n_elements = len(element_names)
    return element_names, n_elements


def create_lookups(element_names):
    """
    Create dictionary to map notes and durations to integers
    :param element_names:
    :return:
    """
    element_to_int = dict((element, number) for number, element in enumerate(element_names))
    int_to_element = dict((number, element) for number, element in enumerate(element_names))

    return element_to_int, int_to_element


def prepare_sequences(notes, durations, lookups, distincts, seq_len =32):
    """ Prepare the sequences used to train the Neural Network """

    note_to_int, int_to_note, duration_to_int, int_to_duration = lookups
    note_names, n_notes, duration_names, n_durations = distincts

    notes_network_input = []
    notes_network_output = []
    durations_network_input = []
    durations_network_output = []

    # create input sequences and the corresponding outputs
    for i in range(len(notes) - seq_len):
        notes_sequence_in = notes[i:i + seq_len]
        notes_sequence_out = notes[i + seq_len]
        notes_network_input.append([note_to_int[char] for char in notes_sequence_in])
        notes_network_output.append(note_to_int[notes_sequence_out])

        durations_sequence_in = durations[i:i + seq_len]
        durations_sequence_out = durations[i + seq_len]
        durations_network_input.append([duration_to_int[char] for char in durations_sequence_in])
        durations_network_output.append(duration_to_int[durations_sequence_out])

    n_patterns = len(notes_network_input)

    # reshape the input into a format compatible with LSTM layers
    notes_network_input = np.reshape(notes_network_input, (n_patterns, seq_len))
    durations_network_input = np.reshape(durations_network_input, (n_patterns, seq_len))
    network_input = [notes_network_input, durations_network_input]

    notes_network_output = np_utils.to_categorical(notes_network_output, num_classes=n_notes)
    durations_network_output = np_utils.to_categorical(durations_network_output, num_classes=n_durations)
    network_output = [notes_network_output, durations_network_output]

    return network_input, network_output


if __name__ == '__main__':
    prepare_data(section='compose',
                 run_id='0007',
                 music_name='cello',
                 seq_len=32)
