import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from ast import literal_eval

from sklearn.preprocessing import StandardScaler


def train_test_split( data, test_subjects=None, test_gestures=None ):
    """
    Split the given dataframe into test and train data and labels. Data is
    of shape (length, 3) where length is the number of samples in a gesture
    reading (can be arbitrary).

    :param data: Dataframe with at least x, y, z accelerometer data and gesture
    index number.

    :param test_subjects: List of user indexes over [0, 7] to select for test
    data. If None, use test_gestures to split.

    :param test_gestures: List of gesture indexes over [0, 19] to select for
    test data. Only used if test_subjects is None.

    :return: Training data, training labels, testing data, testing labels.
    """

    if test_gestures is None and test_subjects is None:
        raise ValueError( "Test subjects must be provided if train ratio is None." )

    column, select_vals = ('user', test_subjects) if test_subjects is not None \
        else ('gesture', test_gestures)


    # Split into train and test rows by selecting rows where slected column data
    # is in the provided splitting list (user or gesture).
    test = data[ data[ column ].isin( select_vals ) ]
    train = data[ ~data[ column ].isin( select_vals ) ]


    # Transpose the data so that the shape is
    # (num_samples, sample_length, num_features).
    # For user 6, 7 test, training data is (2450, 19, 3) assuming we
    # truncate length to 19. Test would be (851, 19, 3).
    X_train = np.array( [ np.array( train[ col ].tolist() ).T for col in [ 'x', 'y', 'z' ] ] ).T
    y_train = np.array( train[ 'gesture' ].tolist() )

    X_test = np.array( [ np.array( test[ col ].tolist() ).T for col in [ 'x', 'y', 'z' ] ] ).T
    y_test = np.array( test[ 'gesture' ].tolist() )

    print( f"{X_train.shape}, {y_train.shape}\n{X_test.shape}, {y_test.shape}" )

    return X_train, y_train, X_test, y_test


def get_gesture_data( path, data_column_headers=None ):
    """
    Load the gesture data from a csv located at 'path'.

    :param path: Location of the gesture data csv file.

    :param data_column_headers: List of data column header names that
    need literal_eval to parse (i.e., the headers for data arrays). If None,
    default to [ 'x', 'y', 'z' ]

    :return: The pandas dataframe holding the gesture data.
    """

    if data_column_headers is None:
        data_column_headers = [ 'x', 'y', 'z' ]

    # Create converter dictionary to literal_eval selected columns.
    converters = {
        header: literal_eval for header in data_column_headers
    }

    return pd.read_csv( path, converters=converters )


def truncate_data( data, length ):
    """
    Truncates x, y, and z data to specified length. If data is less than length,
    it is zero-padded at the end.

    :param data: Pandas dataframe with at least x, y, z columns. Modified
    in place.

    :param length: Length of data to truncate (or pad) to.

    :return: Pandas dataframe with truncated data.
    """
    for idx, row in data.iterrows():
        # If data length is greater than length, truncate.
        # If data length is less than length, zero-pad at end.
        if len( row[ 'x' ] ) > length:
            data.at[ idx, 'x' ] = row[ 'x' ][ :length ]
            data.at[ idx, 'y' ] = row[ 'y' ][ :length ]
            data.at[ idx, 'z' ] = row[ 'z' ][ :length ]
        elif len( row[ 'x' ] ) < length:
            pad_length = length - len( row[ 'x' ] )

            data.at[ idx, 'x' ] = row[ 'x' ][ : ] + ( [ 0 ] * pad_length )
            data.at[ idx, 'y' ] = row[ 'y' ][ : ] + ( [ 0 ] * pad_length )
            data.at[ idx, 'z' ] = row[ 'z' ][ : ] + ( [ 0 ] * pad_length )

    return data


def process_data( path='./aggregated_gesture_data.csv',
                  target='./processed_gesture_data.csv',
                  dir_path=None ):
    """
    Take aggregated gesture data and pre-process it. Use StandardScaler to get
    zero mean and unit variance for x, y, z accelerometer data (separately) for
    each user-gesture iteration.

    :param path: Location of aggregated data csv. If None, provide dir_path
    to top-level directory of the dataset to first accumulate and aggregate
    data.

    :param target: Location to save data frame to. If None, don't save.

    :param dir_path: If path is None, use dir_path to locate the original
    dataset contents and call aggregate_data first.

    :return: Pre-processed data frame.
    """

    # Ensure csv or dataset path is provided.
    if path is None and dir_path is None:
        raise ValueError( "CSV or dataset directory path must be provided." )

    # Get the aggregated data first.
    # Each row is a single gesture attempt and contains a list of the x, y, and
    # z accelerometer data. We can just directly use this dataframe when doing
    # the scaling, so we don't need to copy it or create a new one for the
    # processed data.
    if path is not None:
        aggregated_data = get_gesture_data( path=path )
    else:
        aggregated_data = aggregate_data( path=path, target=None,
                                          dir_path=dir_path )

    # Iterate through each user-gesture-attempt and scale the x, y, and z
    # accelerometer data using sklearn's StandardScaler to remove mean and scale
    # to unit variance.
    scaler = StandardScaler()
    for idx, row in aggregated_data.iterrows():
        # Transpose the data before and after scaling since StandardScaler
        # operates on columns, not rows.
        scaled_data = scaler.fit_transform(
            np.array( [ row[ 'x' ], row[ 'y' ], row[ 'z' ] ] ).T
        ).T

        # Update the data. Convert to list so that it can be properly
        # read in using literal_eval. Otherwise it saves as a NumPy array
        # and that just doesn't work right.
        aggregated_data.at[ idx, 'x' ] = scaled_data[ 0 ].tolist()
        aggregated_data.at[ idx, 'y' ] = scaled_data[ 1 ].tolist()
        aggregated_data.at[ idx, 'z' ] = scaled_data[ 2 ].tolist()

    if target is not None:
        aggregated_data.to_csv( target, index=False )

    return aggregated_data


def aggregate_data( path='./gesture_data.csv',
                    target='./aggregated_gesture_data.csv',
                    dir_path=None ):
    """
    Aggregate gesture data so that each row of the data frame is a single
    gesture attempt (i.e., remove the time stamp information). If path is None,
    then dir_path must be provided so data can be first accumulated and then
    aggregated. Returns data frame. If target is None, data is not saved.

    :param path: Location of accumulated data csv. If None, provide dir_path
    to top-level directory of the dataset to first accumulate data.

    :param target: Location to save data frame to. If None, don't save.

    :param dir_path: If path is None, use dir_path to locate the original
    dataset contents and call accumulate_data first.

    :return: Data frame of aggregated data.
    """

    # Ensure csv or dataset path is provided.
    if path is None and dir_path is None:
        raise ValueError( "CSV or dataset directory path must be provided." )

    # Accumulate data first if csv not provided, otherwise read the csv.
    if path is None and dir_path is not None:
        accumulated_data = accumulate_data( dir_path, target=None,
                                            keep_time_data=False )
    else:
        accumulated_data = pd.read_csv( path, index_col=None )

    # Create an empty data frame to hold the aggregated data.
    # Hold user, gesture, and iteration index along with x, y, z
    # acceleration data for current user-gesture attempt.
    aggregated_data = pd.DataFrame( columns=[ 'user', 'gesture', 'iteration',
                                              'x', 'y', 'z' ] )

    # Configure total number of users and gestures.
    users = 8
    gestures = 20

    # Track the total number of iterations (gesture attempts)
    # and samples (data captures during a gesture attempt).
    total_iterations = 0
    total_samples = 0

    # Iterate through each user-gesture and aggregate the data for each
    # gesture attempt into one row in the aggredated data data frame.
    for user in range( users ):
        for gesture in range( gestures ):
            # Get number of attempts for user-gesture.
            # Select from the accumulated data for all rows with current
            # user-gesture and get the max iteration index. Add one since
            # it is a zero-based index.
            num_iterations = accumulated_data[
                ( accumulated_data[ 'user' ] == user ) &
                ( accumulated_data[ 'gesture' ] == gesture )
            ][ 'iteration' ].max() + 1

            total_iterations += num_iterations

            # Count the number of data collections for the iterations.
            samples = 0

            # Iterate through each iteration and accumulate the readings
            # into a single row for each iteration.
            for iteration in range( num_iterations ):
                # Select the acclerometer data columns from the accumulated
                # data for the current user-gesture iteration. Transpose data
                # to get three rows for x, y, z.
                curr_data = accumulated_data[
                    ( accumulated_data[ 'user' ] == user ) &
                    ( accumulated_data[ 'gesture' ] == gesture ) &
                    ( accumulated_data[ 'iteration' ] == iteration )
                ].loc[ :, [ 'accel0', 'accel1', 'accel2' ] ].values.T

                data = [
                    user, gesture, iteration,
                    curr_data[ 0 ].tolist(), curr_data[ 1 ].tolist(),
                    curr_data[ 2 ].tolist()
                ]

                # Attach the data into the dataframe.
                aggregated_data.loc[ len( aggregated_data.index ) ] = data
                samples += len( curr_data[ 0 ] )
            total_samples += samples


    print(  f"{total_samples = }\n"
            f"{total_iterations = }\n\n"
            f"{total_samples / total_iterations = }" )

    if target is not None:
        aggregated_data.to_csv( target, index=False )

    return aggregated_data



def accumulate_data( dir_path, target='./gesture_data.csv', keep_time_data=True ):
    """
    Provide the path to the top-level directory of the dataset
    and iterate through the dataset to aggregate data from text
    files into a pandas dataframe. Target identifies a path to
    save the accumulated csv data to.

    Resulting csv column headers: user, gesture, iteration,
    millis, nano, timestamp, accel0, accel1, accel2. One row for
    X, Y, Z accelerometer sample.

    :param dir_path: Top-level directory of the dataset.
    :param target: Path to save accumulated data to. If None, don't save.
    :param keep_time_data: Boolean. True to keep the three time data columns.

    :return: Pandas dataframe of the data.
    """

    # Configure total number of users and gestures.
    users = 8
    gestures = 20

    # Initialize empty 2D array.
    # Each row represents a user and holds the paths of the
    # gesture directories.
    directory_list = np.empty( ( users, gestures ), dtype=object )

    # Track the total number of iterations (gesture attempts)
    # and samples (data captures during a gesture attempt).
    total_iterations = 0
    total_samples = 0

    # Iterate through each user and collect the paths to gesture data.
    for i in range( 1, users + 1 ):
        path = f"{dir_path}/U0{i}/"

        directories = []
        for filename in os.listdir( path ):
            f = os.path.join( path, filename )

            if os.path.isdir( f ):
                directories.append( f )

        directories = np.sort( np.array( directories ) )
        directory_list[ i - 1 ] = directories

    df = pd.DataFrame( columns=[ 'user', 'gesture', 'iteration',
                                 'millis', 'nano', 'timestamp',
                                 'accel0', 'accel1', 'accel2' ] )

    for user in range( users ):
        for gesture in range( gestures ):
            # Identify each file in the gesture folder
            # and add it to the list.
            files = []
            for filename in os.listdir( directory_list[ user ][ gesture ] ):
                path = os.path.join( directory_list[ user ][ gesture ], filename )
                if os.path.isfile( path ):
                    files.append( path )

            # Sort the files by gesture order.
            files = np.sort( np.array( files ) )

            # Iterate through each file for the current user-gesture.
            # Count the number of iterations for the user-gesture
            iterations = 0
            for path in files:
                # Count the number of data collections for the iteration.
                samples = 0

                # Iterate through each line (one sample) in the file
                # and push the data to the csv.
                for line in open( path ):
                    # Split the current line on space and parse the data
                    # into the current row of the dataset.
                    data = line.split()
                    if keep_time_data:
                        data = [
                            user, gesture, iterations,
                            int( data[ 0 ] ), int( data[ 1 ] ), int( data[ 2 ] ),
                            float( data[ 3 ] ), float( data[ 4 ] ), float( data[ 5 ] )
                        ]
                    else:
                        data = [
                            user, gesture, iterations,
                            float( data[ 3 ] ), float( data[ 4 ] ), float( data[ 5 ] )
                        ]

                    # Attach the data into the dataframe.
                    df.loc[ len( df.index ) ] = data
                    samples += 1

                total_samples += samples
                iterations += 1

            total_iterations += iterations

    print(  f"{total_samples = }\n"
            f"{total_iterations = }\n\n"
            f"{total_samples / total_iterations = }" )

    if keep_time_data:
        convert_dict = { 'user' : 'int', 'gesture' : 'int', 'iteration' : 'int',
                         'millis' : 'int', 'nano' : 'int', 'timestamp' : 'int' }
    else:
        convert_dict = { 'user' : 'int', 'gesture' : 'int', 'iteration' : 'int' }

    # Ensure data is integer. Likes to default to float.
    df = df.astype( convert_dict )

    print( df.describe() )

    if target is not None:
        df.to_csv( target, index=False )

    return df


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Provided by: https://raw.githubusercontent.com/DTrimarchi10/confusion_matrix/master/cf_matrix.py

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.

    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)