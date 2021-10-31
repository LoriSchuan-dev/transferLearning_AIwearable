import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from ast import literal_eval

from sklearn.preprocessing import StandardScaler


def split_gesture_data( data: pd.DataFrame, gestures: list = None ):
    """
    Split the provided gesture data by gestures. Returns a tuple of two
    pandas DataFrames -- first contains selected gestures, second is the
    holdout gestures.

    :param data: Pandas dataframe of gesture data. Must have at least column
    labeled 'gesture'.

    :param gestures: List of selected gestures as zero-indexed integers.

    :return: Returns a tuple of pandas DataFrames. First is selected gestures
    and second is holdout gestures.
    """

    # Raise a value error if the gestures list is None or empty.
    if not gestures:
        raise ValueError( "Gesture selection must not be None" )

    sel_column = 'gesture'

    selected = data[ data[ sel_column ].isin( gestures ) ]
    holdout = data[ ~data[ sel_column ].isin( gestures) ]

    return selected, holdout


def train_test_split( data, train_subjects=None,
                      test_subjects=None,
                      val_subjects=None,
                      sel_gestures=None ):
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

    :return: Training data, training labels, testing data, testing labels,
    validation data, validation labels.
    """

    if train_subjects is None or test_subjects is None:
        raise ValueError( "Must provide train and test subjects list." )

    # TODO: Select sub-set of gestures if provided.


    # Define the column we're selecting by.
    # Not useful now, maybe in future updates for more dynamic splitting.
    sel_column = 'user'

    val_subjects = [] if val_subjects is None else val_subjects


    # Split into train and test rows by selecting rows where slected column data
    # is in the provided splitting list (user or gesture).
    test = data[ data[ sel_column ].isin( test_subjects ) ]
    train = data[ data[ sel_column ].isin( train_subjects ) ]


    # Transpose the data so that the shape is
    # (num_samples, sample_length, num_features).
    # For user 6, 7 test, training data is (2450, 19, 3) assuming we
    # truncate length to 19. Test would be (851, 19, 3).
    X_train = np.array( [ np.array( train[ col ].tolist() ).T for col in [ 'x', 'y', 'z' ] ] ).T
    y_train = np.array( train[ 'gesture' ].tolist() )

    X_test = np.array( [ np.array( test[ col ].tolist() ).T for col in [ 'x', 'y', 'z' ] ] ).T
    y_test = np.array( test[ 'gesture' ].tolist() )

    # Shuffle the data.
    train_perm = np.random.permutation( X_train.shape[ 0 ] )
    X_train = X_train[ train_perm ]
    y_train = y_train[ train_perm ]

    test_perm = np.random.permutation( X_test.shape[ 0 ] )
    X_test = X_test[ test_perm ]
    y_test = y_test[ test_perm ]

    if val_subjects is not None:
        val = data[ data[ sel_column ].isin( val_subjects ) ]
        X_val = np.array( [ np.array( val[ col ].tolist() ).T for col in [ 'x', 'y', 'z' ] ] ).T
        y_val = np.array( val[ 'gesture' ].tolist() )

        val_perm = np.random.permutation( X_val.shape[ 0 ] )
        X_val = X_val[ val_perm ]
        y_val = y_val[ val_perm ]
    else:
        X_val = None
        y_val = None

    return X_train, y_train, X_test, y_test, X_val, y_val


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

    plt.show()


def print_results_table( scores, data_sels, cf_matrix ):
    """
    Print the results for a number of simulation iterations in a tabular format.

    :param scores: List of tuples from model.evaluate containing test loss and
    test accuracy.

    :param data_sels: List of data set splits at each iteration containing
    training, validation, and test set selections. Each row should be a tuple or
    list of the format (train set, validation set, test set)

    :param cf_matrix: Confusion matrix object from
    sklearn.metrics.confusion_matrix. Used to calculate overall accuracy over
    multiple iterations from the simulation.

    :return: None.
    """

    # Calculate maximum lengths for each of the three data selection sets so 
    # that we can properly set the widths of the train, validation, and test set
    # columns. Creates a three-tuple by finding the max length of the string
    # representation of the lists for train, validation, and test sets at each
    # iteration of the simulation (row of the scores list). If the max length
    # value for a specific column is less than the length of the column header,
    # use the length of the column header instead.
    # First index is train, second is validation, third is test.
    col_headers = [ 'Train Set', 'Validation Set', 'Test Set' ]
    max_lengths = tuple( 
            max( list( map( lambda x: len( str( x[ i ] ) ), data_sels ) ) )
        for i in range( 3 ) )
    max_lengths = [ length if len( col_headers[ i ] ) < length
                    else len( col_headers[ i ] ) 
                    for i, length in enumerate( max_lengths ) ]

    # Pre calculate the strings for the horizontal dividers for the train,
    # validation, and test sets so we're not calculating them at each iteration.
    set_bars = [ '─' * length for length in max_lengths ]


    # Set up widths for the index, accuracy, and loss sections.
    # Does not include single spaces on left and right between vertical pipes.
    index_width = 5
    index_bars = '─' * index_width

    accuracy_width = 8
    accuracy_bars = '─' * accuracy_width

    loss_width = 6
    loss_bars = '─' * loss_width


    # Calculate table width (minus two spaces on either end for spacing)
    table_width = index_width + 3 + accuracy_width + 3 + loss_width + 3 + \
                  max_lengths[ 0 ] + 3 + max_lengths[ 1 ] + 3 + max_lengths[ 2 ]


    # Define outer box to go around the table. Makes it nice and clear.
    # Value is equal to top/bottom spacing. Horizontal spacing is double
    # (monospace is more or less 2:1, width:height).
    spacing = 2
    side_spacing = ' ' * 2 * spacing


    # Calculate overall table width including inner table, two inner padding
    # spaces, inner table borders (2), and padding on both sides.
    total_table_width = table_width + 2 + 2 + ( 2 * 2 * spacing )


    # Print top-side outer box.
    for _ in range( spacing ):
        print()
    print( f"{side_spacing}╔{'═' * total_table_width}╗" )
    for _ in range( spacing ):
        print( f"{side_spacing}║{' ' * total_table_width}║" )


    # Define the inner divider lines that go above each output row.
    inner_divider = f"\n{side_spacing}║{side_spacing}╟─{index_bars}─┼─{accuracy_bars}─" + \
                    f"┼─{loss_bars}─┼─{set_bars[ 0 ]}─┼─{set_bars[ 1 ]}─" + \
                    f"┼─{set_bars[ 2 ]}─╢{side_spacing}║\n"


    # Print the divider and header information
    print( f"{side_spacing}║{side_spacing}╔═{index_bars.replace( '─', '═' )}═"
           f"╤═{accuracy_bars.replace( '─', '═' )}═"
           f"╤═{loss_bars.replace( '─', '═' )}═"
           f"╤═{set_bars[ 0 ].replace( '─', '═' )}═"
           f"╤═{set_bars[ 1 ].replace( '─', '═' )}═"
           f"╤═{set_bars[ 2 ].replace( '─', '═' )}═╗{side_spacing}║\n"
           f"{side_spacing}║{side_spacing}║ {'Index':>{index_width}} "
           f"│ {'Accuracy':>{accuracy_width}} "
           f"│ {'Loss':>{loss_width}} "
           f"│ {col_headers[ 0 ]:^{max_lengths[ 0 ]}} "
           f"│ {col_headers[ 1 ]:^{max_lengths[ 1 ]}} "
           f"│ {col_headers[ 2 ]:^{max_lengths[ 2 ]}} ║{side_spacing}║", end='' )

    # Iterate through each simulation result and print the data in the row.
    for idx, score in enumerate( scores ):
        # train_str = 

        print( f"{inner_divider}"
               f"{side_spacing}║{side_spacing}║ {idx + 1:>{index_width}d} "
               f"│ {score[ 1 ]:>{accuracy_width}.3f} "
               f"│ {score[ 0 ]:>{loss_width}.3f} "
               f"│ {str(data_sels[ idx ][ 0 ]):^{max_lengths[ 0 ]}} "
               f"│ {str(data_sels[ idx ][ 1 ]):^{max_lengths[ 1 ]}} "
               f"│ {str(data_sels[ idx ][ 2 ]):^{max_lengths[ 2 ]}} ║"
               f"{side_spacing}║", 
               end='' )


    # Prin the border for the bottom of the regular data.
    print( f"\n{side_spacing}║{side_spacing}╠═{index_bars.replace( '─', '═' )}═"
           f"╧═{accuracy_bars.replace( '─', '═' )}═"
           f"╧═{loss_bars.replace( '─', '═' )}═"
           f"╧═{set_bars[ 0 ].replace( '─', '═' )}═"
           f"╧═{set_bars[ 1 ].replace( '─', '═' )}═"
           f"╧═{set_bars[ 2 ].replace( '─', '═' )}═╣{side_spacing}║" )


    # Calculate the overall accuracy and total width of the table and print
    # in its own row at the bottom of the table.
    overall_accuracy = 100 * np.trace( cf_matrix ) / float( np.sum( cf_matrix ) )
    print( f"{side_spacing}║{side_spacing}║ {' ' * table_width} ║{side_spacing}║\n"
           f"{side_spacing}║{side_spacing}║ {f'Total Accuracy: {overall_accuracy:{accuracy_width-1}.2f}%':^{table_width}} "
           f"║{side_spacing}║\n"
           f"{side_spacing}║{side_spacing}║ {' ' * table_width} ║{side_spacing}║" )


    # Print the bottom table outline.
    print( f"{side_spacing}║{side_spacing}╚═{'═' * table_width}═╝{side_spacing}║" )


    # Print bottom-side outer box.
    for _ in range( spacing ):
        print( f"{side_spacing}║{' ' * total_table_width}║" )
    print( f"{side_spacing}╚{'═' * total_table_width}╝" )
    for _ in range( spacing ):
        print()


def save_results( scores, data_sels, cf_matrix, epochs, filename, loc='./', 
                  filetype=0 ):
    """
    Save results from model training over a number of iterations to local file
    system as selected filetype (numpy or csv). Confusion matrix cf_matrix will
    be saved separately with '_cf' appended to filename.

    If saving as DataFrame, columns
    are: Loss, Accuracy, Epochs, Train Set, Validation Set, Test Set, Confusion
    Matrix. 

    If saving as a NumPy object, each row contains Loss, Accuracy, Epochs, Train
    Set, Validation Set, Test Set, and Confusion Matrix.

    Loss and accuracy are floats, epochs is an int, 
    train/validation/test sets are lists, and confusion matrix is a list of
    lists.


    :param scores: List of tuples from model.evaluate containing test loss and
    test accuracy.

    :param data_sels: List of data set splits at each iteration containing
    training, validation, and test set selections. Each row should be a tuple or
    list of the format (train set, validation set, test set)

    :param cf_matrix: Confusion matrix object from
    sklearn.metrics.confusion_matrix.

    :param epochs: Number of epochs each iteration was trained on. Should be a
    list of epochs for each iteration. Good for comparing accuracy results to
    number of epochs trained while keeping data_sels the same.

    :param filename: Name of file to save as. File name must not end with a file
    type designator. 

    :param loc: Directory to save data to.

    :param filetype: Type of file to save as. 0 is csv (pandas DataFrame), 1 is
    npz (NumPy)

    :return: Tuple of simulation data and confusion matrix as either DataFrame
    or NumPy object (filetype 0 and 1, respectively).
    """

    # Verify appropriate file type selection.
    if filetype != 0 and filetype != 1:
        raise ValueError( "File type must be 0 (csv) or 1 (npz)." )

    # Check if epochs is a list or an int. If int, convert to list of all 
    # the same values
    if type( epochs ) is not list and type( epochs ) is not int:
        raise TypeError( "Epochs must be list or integer." )
    elif type( epochs ) is not list:
        epochs = [ epochs ] * len( scores )

    # If the directory doesn't end with a slash, add one. Otherwise, do nothing.
    if not loc.endswith( '/' ):
        loc += '/'

    # Check if the save directory exists and make it if not.
    if not os.path.isdir( loc ):
        print( f"'{loc}' does not exist, creating directories..." )
        os.makedirs( loc )

    # Calculate full file name to save to.
    filename = loc + filename

    print( f"Saving results to '{filename}'..." )


    if filetype == 0:
        # pandas DataFrame
        cf_filename = f"{filename}_cf.csv"
        filename = f"{filename}.csv"

        data = {
            'Loss' :            [ row[ 0 ] for row in scores ],
            'Accuracy' :        [ row[ 1 ] for row in scores ],
            'Epochs' :          epochs,
            'Train Set' :       [ row[ 0 ] for row in data_sels ],
            'Validation Set' :  [ row[ 1 ] for row in data_sels ],
            'Test Set' :        [ row[ 2 ] for row in data_sels ]
        }

        cf_data = {
            'Confusion Matrix' : [ row.tolist() for row in cf_matrix ]
        }

        data = pd.DataFrame( data )
        cf_data = pd.DataFrame( cf_data )

        data.to_csv( filename, index=False )
        cf_data.to_csv( cf_filename, index=False )

        print( f"DataFrame for simulation results saved to '{filename}'.\n"
               f"DataFrame for confusion matrix saved to '{cf_filename}'." )
    elif filetype == 1:
        cf_filename = f"{filename}_cf"

        data = np.array( [
            [ row[ 0 ] for row in scores ],
            [ row[ 1 ] for row in scores ],
            epochs,
            [ row[ 0 ] for row in data_sels ],
            [ row[ 1 ] for row in data_sels ],
            [ row[ 2 ] for row in data_sels ],
            [ row.tolist() for row in cf_matrix ]
        ] )

        cf_data = cf_matrix

        np.savez( filename, data )
        np.savez( cf_filename, cf_data )

        print( f"NumPy object for simulation results saved to '{filename}'.\n"
               f"NumPy object for confusion matrix saved to '{cf_filename}." )


    return data, cf_data
