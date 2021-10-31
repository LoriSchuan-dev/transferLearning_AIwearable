import tensorflow as tf
import numpy as np

from util import train_test_split


def create_lstm_model( num_classes=20,
                       dropout=0.8, units=128,
                       data_length=19,
                       optimizer=tf.keras.optimizers.Adam( learning_rate=0.001 )
                    ):
    """
    Create a simple bidirectional LSTM model for classification. Creates a
    bidirectional LSTM layer, dropout, and two dense layers.

    :param num_classes: Number of classes.

    :param dropout: Dropout rate

    :param units: Number of units in LSTM layer.

    :param data_length: Length of time data for LSTM layer input.

    :param optimizer: Optimizer to use for model compilation.

    :return: The compiled model with LSTM, dropout, and two dense layers.
    """

    # Create the model object.
    model = tf.keras.models.Sequential()

    # Add an LSTM layer.
    # Input size is (19,3):
    #   19 time samples from data.
    #   3 dimensions (x, y, z accelerometer data).
    model.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM( units=units, input_shape=[data_length, 3] )
        )
    )

    # Add dropout layer to reduce overfitting.
    model.add( tf.keras.layers.Dropout( rate=dropout ) )

    # Add final dense layers.
    model.add( tf.keras.layers.Dense( units=units, activation='relu' ) )
    model.add( tf.keras.layers.Dense( units=num_classes,
                                      activation='softmax' ) )

    model.compile( loss='sparse_categorical_crossentropy', optimizer=optimizer,
                   metrics=['accuracy'] )

    return model


def train_model( idx, data, model, fit_params,
                 num_classes=20, num_subjects=8, verbose=2 ):
    """
    Wrapper for training and testing model. Returns the testing loss and
    accuracy and the test and predicted labels for the current iteration.

    :param idx: Current iteration index.

    :param data: Pandas dataframe of gesture data.

    :param model: tf.keras model object to train on.

    :param fit_params: Tupled collection of fit/train parameters.
    Contains: epochs, lstm_callbacks.

    :param num_classes: Number of gesture classes to train on.

    :param num_subjects: Number of users in the dataset.

    :param verbose: Verbosity of output.
    0 -- no output. 1 -- Only current iteration output. 2 -- Full.

    :return: Returns tuple of test loss/accuracy, test labels, and predicted
    labels.

    """

    # Unpack model fit parameters.
    epochs, callbacks = fit_params

    # Select the training, test, and validation subjects.
    # Get random ordering of subjects.
    subject_list = np.random.permutation( num_subjects )

    # Select the second from last as validation and last user as test.
    train_subjects = subject_list[ :-3 ].tolist()
    test_subjects = subject_list[ -3:-1 ].tolist()
    val_subjects = [ subject_list[ -1 ] ]


    if verbose > 0:
        print( f"============================================================\n"
               f"Iteration {idx+1}:\n"
               f"    Train Subjects:      {train_subjects}\n"
               f"    Validation Subjects: {val_subjects}\n"
               f"    Test Subjects:       {test_subjects}\n" )

    # Split the data into training, testing, and validation data and labels.
    X_train, y_train, \
    X_test, y_test,\
    X_val, y_val = train_test_split( data,
                                     train_subjects=train_subjects,
                                     test_subjects=test_subjects,
                                     val_subjects=val_subjects )
    validation_data = (X_val, y_val) if X_val is not None else None

    # Train the model on the training data.
    fit_verbose = 0 if verbose <= 1 else 2 if verbose == 2 else 1
    model.fit( X_train, y_train, epochs=epochs, callbacks=callbacks,
               verbose=verbose, validation_data=validation_data )

    # Test the model to see how well we did.
    score = model.evaluate( X_test, y_test )


    return score, y_test, np.argmax( model.predict( X_test ), axis=1 ), \
           ( train_subjects, val_subjects, test_subjects )
