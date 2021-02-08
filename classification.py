import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np

# ----------------------------------------------------------------------------------------------------------- #
# Utility APIs
# ----------------------------------------------------------------------------------------------------------- #

# Encoder dictionary containing unique encoders for all columns
attrEncoders = dict()


# This api encodes individual columns of the data
# Unique encoders of each column are added to @attrEncoders
# for decoding or reuse if required
# Note: Missing values in the data are not encoded and are kept as it is
# Input: data as Dataframe,
# refresh = False/True -> use saved/new encoders
# Output: encoded data as Dataframe
def label_encode(encode, refresh=True):
    if refresh is True:
        attrEncoders.clear()

    encoded_data = encode
    for attr in encoded_data.columns:
        column = encoded_data[attr]

        # If encoder for the column is already available, retrieve it.
        # Else create new one
        if attr not in attrEncoders:
            encoder = LabelEncoder()
        else:
            encoder = attrEncoders[attr]

        # Encode the data leaving all missing values as it is.
        encoded_data[attr] = pd.Series(
            encoder.fit_transform(column[column.notnull()]),
            index=column[column.notnull()].index
        )
        # Add corresponding encoder to dictionary
        attrEncoders[attr] = encoder
    return encoded_data


# This api decodes individual columns of the data
# Note: Missing values in the data are not decoded and are kept as it is
# Input: data as Dataframe
# Output: decoded data as Dataframe
def label_decode(decode):
    decoded_data = decode
    for attr in decoded_data.columns:
        column = decoded_data[attr]

        # If encoder for the column is already available, retrieve it.
        # Else throw error
        if attr not in attrEncoders:
            print("Error. No encoder found for column:", attr)
            exit(1)
        else:
            encoder = attrEncoders[attr]

        # Decode the data leaving all missing values as it is.
        decoded_data[attr] = pd.Series(
            encoder.inverse_transform(column[column.notna()].astype(dtype='int32')),
            index=column[column.notnull()].index
        )
    return decoded_data


# A decision tree classifier with default parameters
# Classifies the data over training set
# Predicts the output of test set
# Returns the error rate by comparing prediction with actual output
# Input: Training and Test sets
# Output: Error rate, accuracy score
def decision_tree_error(X_train, y_train, X_test, y_test):
    # Create, fit through Decision tree classifier
    # with default parameters
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    # Get predicted values over test set
    y_hat = classifier.predict(X_test)

    # Compare the predicted output with actual output of test set
    # and output the error rate
    count = 0
    for i in range(0, y_test.shape[0]):
        if y_hat[i] == y_test[i]:
            count += 1
    score = (count / X_test.shape[0])
    return 1 - score


# ----------------------------------------------------------------------------------------------------------- #
# Reading, separating and refining data
# ----------------------------------------------------------------------------------------------------------- #

# Read data file to a dataframe
adult = pd.read_csv("./data/adult.csv")

# Separate attributes and target columns
data = adult.drop(columns=['class'])
target = adult['class']

# Drop 'fnlwgt' column
data = data.drop(columns=['fnlwgt'])


# ----------------------------------------------------------------------------------------------------------- #
# Q1 This constitutes Question 1 of Classification
# ----------------------------------------------------------------------------------------------------------- #

print("Number of instances:", len(data))

# Find the missing values, sum over each column and then sum them too.
n_missing = data.isnull().sum().sum()
print("Number of missing values:", n_missing)

# Divide above result by all values in data and round result to 4 decimals
print("Fraction of missing values over all attributes:", np.round(n_missing / data.size, 4))

# Check for null values in each instance and add to total if at least one missing value is found
n_inst_missing = (data.isnull().sum(axis=1) > 0).sum()
print("Number of instances with missing values:", n_inst_missing)

# Divide above result with total instances and round result to 3 decimals
print("Fraction of instances with missing values over all instances:", np.round(n_inst_missing / len(data), 3))


# ----------------------------------------------------------------------------------------------------------- #
# Q2 This constitutes Question 2 of Classification
# ----------------------------------------------------------------------------------------------------------- #

# Label the data using LabelEncoder and print all discrete values for each attribute
encoded_data = label_encode(data)

# Get the encoded data and print all discrete values of the column
for attr, encoder in attrEncoders.items():
    print("Discrete values in", attr, encoder.classes_)


# ----------------------------------------------------------------------------------------------------------- #
# Q3 This constitutes Question 3 of Classification
# ----------------------------------------------------------------------------------------------------------- #

# Remove all missing values from encoded data
# and then remove those indexes of removed instances from target data
valid_instances = encoded_data[encoded_data.notnull().all(axis=1)]
valid_target = target[encoded_data.notnull().all(axis=1)]

# Split data to train/test set with 20% data going to test set
X_train, X_test, y_train, y_test = train_test_split(valid_instances.to_numpy(), valid_target.to_numpy(),
                                                    test_size=0.2)
# Get error rate after training and predicting from Decision Tree Classifier
error = decision_tree_error(X_train, y_train, X_test, y_test)
print("Error rate: %.2f" % error)


# ----------------------------------------------------------------------------------------------------------- #
# Q4 This constitutes Question 4 of Classification
# ----------------------------------------------------------------------------------------------------------- #

# Decode data to get original values to start working on missing values
data = label_decode(data)

# Find all the instances with at least one missing value
missing_instances = data[data.isnull().any(axis=1)]
# Find the remaining instances with no missing value
non_missing_instances = data[data.notnull().all(axis=1)]

# Sample data from @non_missing_instances set with same size as @missing_instances set
sample_set = non_missing_instances.sample(n=missing_instances.shape[0], replace=False)
# Combine the two equal sets (one with missing values, one without missing values)
d_dash = missing_instances.append(sample_set)

# Find the remaining instances from data not included in above create set
# This will be used as test set
d_test = data.loc[~data.index.isin(d_dash.index)]

# Split the target based on above sets
d_dash_target = target.loc[target.index.isin(d_dash.index)]
d_test_target = target.loc[target.index.isin(d_test.index)]


# Q4.1 Replace missing values with "missing" value
# -------------------------------------------------------- #

#  Fill all missing values with string value "missing"
d_dash_1 = d_dash.fillna("missing")
# Encode the attributes of training and test data with LabelEncoder
encoded_data = label_encode(d_dash_1.append(d_test))
d_dash_1 = encoded_data.iloc[:len(d_dash_1), :]
d_test_1 = encoded_data.iloc[len(d_dash_1):, :]

# Get error rate after training and predicting from Decision Tree Classifier
error = decision_tree_error(d_dash_1.to_numpy(), d_dash_target.to_numpy(), d_test_1.to_numpy(), d_test_target.to_numpy())
print("Error rate D'1: %.5f" % error)


# Q4.2 Replace missing values with most popular value
# -------------------------------------------------------- #

# Use mode value as fillers
d_dash_2 = d_dash.fillna(data.mode().iloc[0])
# Encode the attributes of training and test data with LabelEncoder
encoded_data = label_encode(d_dash_2.append(d_test))
d_dash_2 = encoded_data.iloc[:len(d_dash_2), :]
d_test_2 = encoded_data.iloc[len(d_dash_2):, :]

# Get error rate after training and predicting from Decision Tree Classifier
error = decision_tree_error(d_dash_2.to_numpy(), d_dash_target.to_numpy(), d_test_2.to_numpy(), d_test_target.to_numpy())
print("Error rate D'2: %.5f" % error)