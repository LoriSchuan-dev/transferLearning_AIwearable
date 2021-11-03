from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

from models import create_lstm_model, train_model


def main():
    ### Define simulation parameters.
    data_truncate_pad_length = 26
    epochs = 50
    num_iterations = 1
    num_gestures = 20
    first_gesture_count = 15
    data_percentage = 1.0
    user_selections = []


    # Define naive model parameters.
    dropout_rate = 0.6
    lstm_units = 48
    lr = 0.005
    lstm_optimizer = Adam( learning_rate=lr, decay=1e-6 )
    monitor = 'loss'
    min_delta = 0.0001
    patience = 3
    earlystop_callback = EarlyStopping( monitor=monitor, min_delta=min_delta,
                                        verbose=1, patience=patience )
    lstm_callbacks = [earlystop_callback]
    # lstm_callbacks = None

    model_params = ( dropout_rate, lstm_units, data_truncate_pad_length, lstm_optimizer )
    fit_params = ( epochs, lstm_callbacks )


    # Load data and trim/pad to set length.
    data_full = get_data( data_length=data_truncate_pad_length )

    # Select the gestures to train the naive model on.
    # Remaining gestures will be trained
    first_gestures = np.arange( first_gesture_count )
    data = data_full[ data_full[ 'gesture' ].isin( first_gestures ) ]


    # Create data structures to hold confusion matrix and loss/accuracy results
    # for each iteration.
    cf_matrix_true = np.array( [] )
    cf_matrix_pred = np.array( [] )
    scores = []


    for i in range( num_iterations ):
        lstm_naive_model = create_lstm_model( num_classes=15,
                                              dropout=dropout_rate,
                                              units=lstm_units,
                                              data_length=data_truncate_pad_length,
                                              optimizer=lstm_optimizer )

        score, y_test, y_pred,\
        train_val_test_splits = train_model( i, data, lstm_naive_model,
                                             fit_params, classes=first_gestures,
                                             verbose=1 )

        user_selections.append( train_val_test_splits )

        scores.append( score )

        # Generate data for confusion matrix.
        cf_matrix_true = np.hstack( ( cf_matrix_true, y_test ) )
        cf_matrix_pred = np.hstack( ( cf_matrix_pred, y_pred ) )

        # Logging output for current iteration.
        print( "test loss, test acc: ", score )

        f_model = Sequential()
        f_model.add( lstm_naive_model )
        f_model.layers.pop()
        # for layer in f_model.layers[ :-1 ]:
        #     layer.trainable = False
        f_model.add( Dense( 20, activation='softmax' ) )
        f_model.compile( loss='categorical_crossentropy',
                         optimizer=lstm_optimizer, metrics=['accuracy'] )

        # train_model( i, data_full, f_model, (epochs//2, lstm_callbacks),
        #              classes=np.arange( gestures ), verbose=1 )
        # f_model.layers[ -2 ].trainable = False
        score, y_test, y_pred, \
        train_val_test_splits = train_model( i, data_full, f_model, fit_params,
                                             classes=np.arange( num_gestures ),
                                             verbose=1 )
        f_model.summary()

        print( "TRANSFER test loss, test acc: ", score )

    # Generate the confusion matrix.
    cf_matrix = confusion_matrix( cf_matrix_true, cf_matrix_pred )


    # Print the results for each simulation run in a tabular format.
    print_results_table( scores, user_selections, cf_matrix )


    # Save results.
    # save_results( scores, user_selections, cf_matrix, epochs,
    #               filename=f'results_{epochs}-epochs_{num_iterations}-iterations', 
    #               loc='./results/', filetype=0 )


    # Plot the confusion matrix.
    make_confusion_matrix( cf_matrix, categories=np.arange( first_gesture_count ),
                           figsize=[8,8])


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
