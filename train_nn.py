import os

from keras.callbacks import ModelCheckpoint, EarlyStopping

from data_preparation import prepare_data

from nn_creation import create_network


def train_network():
    section = 'compose'
    run_id = '0007'
    music_name = 'cello'
    seq_len = 32

    notes, n_notes, durations, n_durations, lookups, distincts, network_input, network_output, run_folder = \
        prepare_data(section=section,
                     run_id=run_id,
                     music_name=music_name,
                     seq_len=seq_len)
    # model params
    embed_size = 100
    rnn_units = 256
    use_attention = True

    model, att_model = create_network(n_notes=n_notes,
                                      n_durations=n_durations,
                                      embed_size=embed_size,
                                      rnn_units=rnn_units,
                                      use_attention=use_attention)

    print(model.summary())

    weights_folder = os.path.join(run_folder, 'weights')

    checkpoint1 = ModelCheckpoint(
        os.path.join(weights_folder, "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.h5"),
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )

    checkpoint2 = ModelCheckpoint(
        os.path.join(weights_folder, "weights.h5"),
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )

    early_stopping = EarlyStopping(
        monitor='loss'
        , restore_best_weights=True
        , patience=10
    )

    callbacks_list = [
        checkpoint1
        , checkpoint2
        , early_stopping
    ]

    model.save_weights(os.path.join(weights_folder, "weights.h5"))
    model.fit(network_input, network_output
              , epochs=50, batch_size=32
              , validation_split=0.2
              , callbacks=callbacks_list
              , shuffle=True
              )


if __name__ == '__main__':
    train_network()
