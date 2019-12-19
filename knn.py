"""Implementation of the K-nearest-neighbors algorithm."""
import csv
import heapq
from functools import partial


def euclidean_distance(example1, example2):
    """Calculate the euclidean distance over two vectors.

    Euclidean distance is the square root of the sum of all squared values between vector1 and vector2.
    d(a,b) = sqrt((a1-b1) ** 2 + (a2-b2) ** 2 + ...)

    Args:
        example1 (list): list of floats represents the first vector.
        example2 (list): list of floats represents the second vector.

    Returns:
        number. the euclidean distance between the two vectors.

    Note:
        This function removes the last value from the vector as it is treated as the label of the item.
    """
    example1_data = example1[:-1]
    example2_data = example2[:-1]
    return sum((float(a1) - float(a2)) ** 2 for a1, a2 in zip(example1_data, example2_data)) ** 0.5


def manhattan_distance(example1, example2):
    """Calculate the manhattan distance over two vectors.

    Manhattan distance is the sum of all subtraction between values from v1 and v2.
    d(a,b) = (a1-b1) + (a2-b2) + ...

    Args:
        example1 (list): list of floats represents the first vector.
        example2 (list): list of floats represents the second vector.

    Returns:
        number. the manhattan distance between the two vectors.

    Note:
        This function removes the last value from the vector as it is treated as the label of the item.
    """
    example1_data = example1[:-1]
    example2_data = example2[:-1]
    return sum(abs(float(a1) - float(a2)) for a1, a2 in zip(example1_data, example2_data))


def hamming_distance(example1, example2):
    """Calculate the hamming distance over two vectors.

    Hamming distance is the amount of changed values between two vectors.

    Args:
        example1 (list): list of floats represents the first vector.
        example2 (list): list of floats represents the second vector.

    Returns:
        number. the hamming distance between the two vectors.

    Note:
        This function removes the last value from the vector as it is treated as the label of the item.
    """
    example1_data = example1[:-1]
    example2_data = example2[:-1]
    return sum(el1 != el2 for el1, el2 in zip(example1_data, example2_data))


def load_set(path):
    """Open and read data set from a csv file.

    Args:
        path (str): the path to the data set file.

    Returns:
        list. list of all examples written in the csv file.

    Note:
        All the examples include their labels.
    """
    with open(path) as set_file:
        return list(csv.reader(set_file))


class Knn:
    """K-Nearest-neighbor classifier class.

    Attributes:
        k (number): the amount of neighbors to classify by.
        distance_func (func): the nearest neighbor distance evaluation function.
    """

    def __init__(self, k=5, distance=euclidean_distance):
        self.k = k
        self.distance_func = distance

    def fit(self, train_set):
        """Train the model.

        This function doesnt execute any calculations since the knn model doesnt need training.

        Args:
            train_set (list): list of the training examples.
        """
        self.examples = train_set

    def classify(self, example):
        """Predict the label of the given example.

        Args:
            example (list): vector of features, the model will try to label.

        Returns:
            str. the model's output label.
        """
        # Embed one argument in the distance function as the current example
        distance = partial(self.distance_func, example)

        # find the k nearest examples
        nearest_neighbors = heapq.nsmallest(self.k, self.examples, key=distance)
        labels = [str(label) for *data, label in nearest_neighbors]

        return max(set(labels), key=labels.count)

    @classmethod
    def n_fold_validation(cls, n, train_set_path, k, distance):
        """Split the training set to train set and a test set and evaluate the accuracy.

        Args:
            n (number): the size of the training set (the rest is test set) e.g. n=0.66
            train_set_path (str): the path to the training set csv file.
            k (number): the amount of neighbors to classify by.
            distance (func): the nearest neighbor distance evaluation function.

        Returns:
            number. the accuracy of the test set classification.
        """
        set = load_set(train_set_path)[1:] # Load the training set and split the titles row

        # Split the training set to train and test set.
        split_index = int(len(set) * n)
        train_set = set[:split_index]
        test_set = set[split_index:]

        # Create and train the classifier
        clf = cls(k, distance)
        clf.fit(train_set)

        # Classify and evaluate the accuracy.
        correct_classification = sum(clf.classify(ex) == ex[-1] for ex in test_set)
        return correct_classification / len(test_set)


def classify_file(file_path, classifier, output_file_path):
    """Label a whole test set.

    Args:
        file_path (str): the path to the test set to label.
        classifier (object): classifier instance.
        output_file_path (str): path to a csv file to write all labeled data to.
    """
    titles, *test_set = load_set(file_path)
    total = len(test_set)
    with open(output_file_path, 'w') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(titles)
        for i, example in enumerate(test_set):
            print(f'{i * 100 // total}%', end='\r')
            label = classifier.classify(example)
            example[-1] = label
            csv_writer.writerow(example)
