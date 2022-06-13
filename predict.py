import os
import pickle

import time

import numpy as np
from music21 import instrument, note, stream, chord, duration

from nn_creation import create_network


def load_lookup_tables(store_folder):
    """
    load lookup tables
    :param store_folder:
    :return: all lookup tables which were created on data_preparation stage
    """
    with open(os.path.join(store_folder, 'distincts'), 'rb') as filepath:
        distincts = pickle.load(filepath)
        note_names, n_notes, duration_names, n_durations = distincts

    with open(os.path.join(store_folder, 'lookups'), 'rb') as filepath:
        lookups = pickle.load(filepath)
        note_to_int, int_to_note, duration_to_int, int_to_duration = lookups

    return note_names, n_notes, duration_names, n_durations, note_to_int, int_to_note, duration_to_int, int_to_duration


def build_model(n_notes,
                n_durations,
                embed_size,
                rnn_units,
                use_attention,
                weights_folder,
                weights_file):
    model, att_model = create_network(n_notes, n_durations, embed_size, rnn_units, use_attention)

    # Load the weights to each node
    weight_source = os.path.join(weights_folder, weights_file)
    model.load_weights(weight_source)
    print(model.summary())

    return model, att_model


def generate_notes(notes,
                   durations,
                   note_to_int,
                   duration_to_int,
                   max_extra_notes,
                   sequence_length,
                   model,
                   att_model,
                   notes_temp,
                   duration_temp):
    prediction_output = []
    notes_input_sequence = []
    durations_input_sequence = []

    overall_preds = []

    for n, d in zip(notes, durations):
        note_int = note_to_int[n]
        duration_int = duration_to_int[d]

        notes_input_sequence.append(note_int)
        durations_input_sequence.append(duration_int)
        prediction_output.append([n, d])

        if n != 'START':
            midi_note = note.Note(n)
            new_note = np.zeros(128)
            new_note[midi_note.pitch.midi] = 1
            overall_preds.append(new_note)

    att_matrix = np.zeros(shape=(max_extra_notes + sequence_length, max_extra_notes))

    for note_index in range(max_extra_notes):
        prediction_input = [
            np.array([notes_input_sequence]),
            np.array([durations_input_sequence])
        ]
        notes_prediction, durations_prediction = model.predict(prediction_input, verbose=0)
        if use_attention:
            att_prediction = att_model.predict(prediction_input, verbose=0)[0]
            att_matrix[(note_index - len(att_prediction) + sequence_length):(note_index + sequence_length),
                       note_index] = att_prediction

        new_note = np.zeros(128)

        for idx, n_i in enumerate(notes_prediction[0]):
            try:
                note_name = int_to_note[idx]
                midi_note = note.Note(note_name)
                new_note[midi_note.pitch.midi] = n_i

            except:
                pass

        overall_preds.append(new_note)
        i1 = sample_with_temp(notes_prediction[0], notes_temp)
        i2 = sample_with_temp(durations_prediction[0], duration_temp)

        note_result = int_to_note[i1]
        duration_result = int_to_duration[i2]

        prediction_output.append([note_result, duration_result])

        notes_input_sequence.append(i1)
        durations_input_sequence.append(i2)

        if len(notes_input_sequence) > max_seq_len:
            notes_input_sequence = notes_input_sequence[1:]
            durations_input_sequence = durations_input_sequence[1:]

        if note_result == 'START':
            break

    overall_preds = np.transpose(np.array(overall_preds))
    print('Generated sequence of {} notes'.format(len(prediction_output)))

    return overall_preds, prediction_output


def sample_with_temp(preds, temperature):

    if temperature == 0:
        return np.argmax(preds)
    else:
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return np.random.choice(len(preds), p=preds)


if __name__ == '__main__':
    # run params
    section = 'compose'
    run_id = '0007'
    music_name = 'cello'
    run_folder = os.path.join('run', section)
    run_folder = os.path.join(run_folder, '_'.join([run_id, music_name]))

    # model params
    embed_size = 100
    rnn_units = 256
    use_attention = True

    store_folder = os.path.join(run_folder, 'store')
    note_names, n_notes, duration_names, n_durations, note_to_int, int_to_note, duration_to_int, int_to_duration = \
        load_lookup_tables(store_folder=store_folder)

    weights_folder = os.path.join(run_folder, 'weights')
    weights_file = 'weights.h5'
    model, att_model = build_model(n_notes=n_notes,
                                   n_durations=n_durations,
                                   embed_size=embed_size,
                                   rnn_units=rnn_units,
                                   use_attention=use_attention,
                                   weights_folder=weights_folder,
                                   weights_file=weights_file)

    # prediction params
    notes_temp = 0.5
    duration_temp = 0.5
    max_extra_notes = 50
    max_seq_len = 32
    seq_len = 32

    # notes = ['START', 'D3', 'D3', 'E3', 'D3', 'G3', 'F#3','D3', 'D3', 'E3', 'D3', 'G3', 'F#3','D3', 'D3', 'E3', 'D3', 'G3', 'F#3','D3', 'D3', 'E3', 'D3', 'G3', 'F#3']
    # durations = [0, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2]

    # notes = ['START', 'F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3','F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3']
    # durations = [0, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2]

    notes = ['START']
    durations = [0]

    if seq_len is not None:
        notes = ['START'] * (seq_len - len(notes)) + notes
        durations = [0] * (seq_len - len(durations)) + durations

    sequence_length = len(notes)

    overall_preds, prediction_output = generate_notes(notes,
                                                      durations,
                                                      note_to_int,
                                                      duration_to_int,
                                                      max_extra_notes,
                                                      sequence_length,
                                                      model,
                                                      att_model,
                                                      notes_temp,
                                                      duration_temp)

    output_folder = os.path.join(run_folder, 'output')

    midi_stream = stream.Stream()

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        note_pattern, duration_pattern = pattern
        # pattern is a chord
        if ('.' in note_pattern):
            notes_in_chord = note_pattern.split('.')
            chord_notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(current_note)
                new_note.duration = duration.Duration(duration_pattern)
                new_note.storedInstrument = instrument.Violoncello()
                chord_notes.append(new_note)
            new_chord = chord.Chord(chord_notes)
            midi_stream.append(new_chord)
        elif note_pattern == 'rest':
            # pattern is a rest
            new_note = note.Rest()
            new_note.duration = duration.Duration(duration_pattern)
            new_note.storedInstrument = instrument.Violoncello()
            midi_stream.append(new_note)
        elif note_pattern != 'START':
            # pattern is a note
            new_note = note.Note(note_pattern)
            new_note.duration = duration.Duration(duration_pattern)
            new_note.storedInstrument = instrument.Violoncello()
            midi_stream.append(new_note)

    midi_stream = midi_stream.chordify()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    midi_stream.write('midi', fp=os.path.join(output_folder, 'output-' + timestr + '.mid'))
