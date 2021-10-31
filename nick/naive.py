from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import numpy as np

from util import make_confusion_matrix
from util import accumulate_data
from util import aggregate_data
from util import process_data
from util import get_gesture_data
from util import truncate_data
from util import train_test_split
from util import print_results_table
from util import save_results

from models import create_lstm_model


def main():
    # Define simulation parameters.
    data_length = 26        # Length to truncate/pad samples to.
    epochs = 100             # Length of training.
    num_iterations = 15     # Number of iterations of fit/test.
    user_selections = []    # List to hold tuple of user splits (train/test/val)


    # Define naive model parameters.
    dropout_rate = 0.8
    lstm_units = 64

    lr = 0.005
    lstm_optimizer = Adam( learning_rate=lr, decay=1e-6 )

    monitor = 'loss'
    min_delta = 0.0001
    patience = 3
    earlystop_callback = EarlyStopping( monitor=monitor, min_delta=min_delta,
                                  verbose=1, patience=patience )
    lstm_callbacks = [earlystop_callback]
    # lstm_callbacks = None

    model_params = ( dropout_rate, lstm_units, data_length, lstm_optimizer )
    fit_params = ( epochs, lstm_callbacks )


    # Load data and trim/pad to set length.
    data = get_data( data_length=data_length )


    # Create data structures to hold confusion matrix and loss/accuracy results
    # for each iteration.
    cf_matrix_true = np.array( [] )
    cf_matrix_pred = np.array( [] )
    scores = []


    for i in range( num_iterations ):
        score, y_test, y_pred,\
        train_val_test_splits = train_model( i, data, model_params, 
                                             fit_params, verbose=1 )

        user_selections.append( train_val_test_splits )

        scores.append( score )

        # Generate data for confusion matrix.
        cf_matrix_true = np.hstack( ( cf_matrix_true, y_test ) )
        cf_matrix_pred = np.hstack( ( cf_matrix_pred, y_pred ) )

        # Logging output for current iteration.
        print( "test loss, test acc: ", score )

    # Generate the confusion matrix.
    # cf_matrix = confusion_matrix( y_test, np.argmax( lstm_naive_model.predict( X_test ), axis=1 ) )
    cf_matrix = confusion_matrix( cf_matrix_true, cf_matrix_pred )


    # Print the results for each simulation run in a tabular format.
    print_results_table( scores, user_selections, cf_matrix )


    # Save results.
    # save_results( scores, user_selections, cf_matrix, epochs,
    #               filename=f'results_{epochs}-epochs_{num_iterations}-iterations', 
    #               loc='./results/', filetype=0 )


    # Plot the confusion matrix.
    # make_confusion_matrix( cf_matrix, categories=[ '01', '02', '03', '04', '05',
    #                                                '06', '07', '08', '09', '10',
    #                                                '11', '12', '13', '14', '15',
    #                                                '16', '17', '18', '19', '20' ],
    #                        figsize=[8,8])


def train_model( idx, data, model_params, fit_params,
                 num_classes=20, num_subjects=8, verbose=2 ):
    """
    Wrapper for training and testing model. Returns the testing loss and
    accuracy and the test and predicted labels for the current iteration.

    :param idx: Current iteration index.

    :param data: Pandas dataframe of gesture data.

    :param model_params: Tupled collection of LSTM model parameters. 
    Contains: dropout_rate, lstm_units, data_length, lstm_optimizer

    :param fit_params: Tupled collection of fit/train parameters.
    Contains: epochs, lstm_callbacks.

    :param num_classes: Number of gesture classes to train on.

    :param num_subjects: Number of users in the dataset.

    :param verbose: Verbosity of output.
    0 -- no output. 1 -- Only current iteration output. 2 -- Full.

    :return: Returns tuple of test loss/accuracy, test labels, and predicted
    labels.

    """

    # Get model parameters
    dropout_rate, lstm_units, data_length, lstm_optimizer = model_params
    epochs, lstm_callbacks = fit_params

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

    # Create the naive LSTM model without dropout.
    lstm_naive_model = create_lstm_model( num_classes=num_classes,
                                          dropout=dropout_rate,
                                          units=lstm_units,
                                          data_length=data_length,
                                          optimizer=lstm_optimizer )

    # Train the model on the training data.
    fit_verbose = 0 if verbose <= 1 else 2 if verbose == 2 else 1
    lstm_naive_model.fit( X_train, y_train, epochs=epochs,
                          callbacks=lstm_callbacks, verbose=verbose,
                          validation_data=validation_data )

    # Test the model to see how well we did.
    score = lstm_naive_model.evaluate( X_test, y_test )


    return score, y_test, \
            np.argmax( lstm_naive_model.predict( X_test ), axis=1 ), \
            ( train_subjects, val_subjects, test_subjects )
            


def get_data( data_length ):
    """
    Process and read in the gesture data.

    :param data_length: Desired length to truncate the accelerometer data to. If
    None, do not truncate/pad data.

    :return: Gesture data
    """

    ############################################################################
    #### Below are examples of how to process the data.
    # Accumulate data from original text files into a single (pandas dataframe).
    # Data frame columns are user, gesture, iteration (attempt number), millis,
    # nano, timestamp, accel0, accel1, accel2.
    # Each row is a separate sample from the accelerometer.
    # data = accumulate_data( '../tev-gestures-dataset/',
    #                           target='./gesture_data.csv' )

    # Take the data from accumulate_data and aggregate the iterations so that
    # each row is a single gesture attempt (iteration). Removes the millis,
    # nano, and timestamp.
    # data = aggregate_data( './gesture_data.csv',
    #                           target='./aggregated_gesture_data.csv',
    #                           dir_path=None )

    # After accumulating the data, scale it so that each accelerometer axis has
    # a zero mean and unit variance. This scaling is done per gesture attempt
    # and per axis (you can test a couple of samples to verify that the mean is
    # approximately zero and the variance is approximately 1).
    # data = process_data( './aggregated_gesture_data.csv',
    #                      target='./processed_gesture_data.csv', dir_path=None )
    ############################################################################


    # Load in the pre-processed data.
    data = get_gesture_data( './processed_gesture_data.csv' )

    # Truncate the data as desired. Comment out to test non-truncated data.
    # Make sure your model can handle variable length data!
    data = truncate_data( data, length=data_length )

    return data


if __name__ == "__main__":
    main()
