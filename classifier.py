"""
CLI part of the project that also wraps the calls to scikit-learn.

Some of the help texts are copied from the scikit-learn documentation
"""

import numpy as np

from sklearn import preprocessing
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

import json
import os

import click
import time

from preprocessing import DB, knn_impute, MI_RNA_NAMES


@click.group()
@click.option("--db", default="db.json", help="The location of the json representation of the database")
@click.pass_context
def cli(ctx, db):
    """
    Tumour Classifier by Johannes Bechberger

    This tool allows using different machine learning algorithms (from scikit-learn)
    on the microRNA data from The Cancer Genome Altas project."""
    ctx.obj = {}
    if os.path.exists(db):
        ctx.obj["db"] = DB.load(db)
    else:
        ctx.obj["db"] = DB({},{})
    ctx.obj["db_file"] = db


@cli.command()
@click.argument("case_ids", nargs=-1)
@click.pass_context
def update_db(ctx, case_ids):
    """Pull individual cases from the TCGA database"""
    try:
        ctx.obj["db"].pull_all_cases(case_ids)
        ctx.obj["db"].store(ctx.obj["db_file"])
    except:
        ctx.obj["db"].store("{}.{}".format(ctx.obj["db_file"], time.time()))
        raise


@cli.command()
@click.argument("case_dict_file", type=click.File())
@click.pass_context
def update_db_from_dict(ctx, case_dict_file):
    """Pull cases from the TCGA database. The case file can be obtained by clicking on the right hand-side button of the table at the cases table view in the data portal.
    Aborting this command leads to "{db file name}.{time stamp}" database file
    """
    try:
        ctx.obj["db"].pull_all_cases_from_dict_list(json.load(case_dict_file))
        ctx.obj["db"].store(ctx.obj["db_file"])
    except:
        ctx.obj["db"].store("{}.{}".format(ctx.obj["db_file"], time.time()))
        raise


@cli.command()
@click.argument("new_db_file", nargs=1)
@click.option("--k", type=int, default=5,
              help="Number of nearest neighbors to consider")
@click.pass_context
def impute(ctx, new_db_file: str, k: int):
    """Impute missing normal sample using the average of the k nearest neighbors"""
    knn_impute(ctx.obj["db"], k)
    ctx.obj["db"].store(new_db_file)

@cli.group()
@click.option("--blood/--no_blood", default=True,
            help="Include blood normal samples in the set of tumour tissue samples")
@click.option("--only_blood/--not_only_blood", default=False,
            help="Use blood normal samples instead of tumour tissue samples")
@click.option("--min_samples", default=20, type=int,
            help="Minimum samples for tumour to be considered")
@click.pass_context
def learn(ctx, blood: bool, only_blood: bool, min_samples: int):
    """Group of commands that work on the database"""
    ctx.obj["blood"] = blood
    ctx.obj["only_blood"] = only_blood
    X, y = ctx.obj["db"].sample_arrs_and_features(blood_normals=blood, tumour=not only_blood,
                                                                             min_samples=min_samples)
    ctx.obj["samples_and_features"] = X, y
    ctx.obj["min_samples"] = min_samples

@learn.command()
@click.pass_context
def stats(ctx):
    """Show some stats about the samples in the database"""
    samples_per_tumour = ctx.obj["db"].samples_per_tumour(ctx.obj["blood"], not ctx.obj["only_blood"])
    used_lines = []
    omitted_lines = []
    for tumour in sorted(samples_per_tumour.keys()):
        line = "    {:50s} {:10d}".format(tumour, samples_per_tumour[tumour])
        if samples_per_tumour[tumour] >= ctx.obj["min_samples"]:
            used_lines.append(line)
        else:
            omitted_lines.append(line)
    if len(used_lines):
        print("--- Samples per tumour (for {} tumours with enough samples)".format(len(used_lines)))
        for line in used_lines:
            print(line)
    if len(omitted_lines):
        print("--- {} Tumours which are omitted, because they have less than {} samples"
              .format(len(omitted_lines), ctx.obj["min_samples"]))
        for line in omitted_lines:
            print(line)
    print("{} normal samples".format(ctx.obj["db"].normal_sample_count()))
    print("Overall {} cases and {} samples".format(len(ctx.obj["db"].cases), len(ctx.obj["db"].samples)))

def store_samples_and_features(X: np.array, y, filename: str):
    with open(filename, "w") as f:
        json.dump({"X": X.tolist(), "y": y}, f)


def load_samples_and_features(filename: str) -> tuple:
    with open(filename, "r") as f:
        d = json.load(f)
        return d["X"], d["y"]


@learn.command()
@click.option("--output_file", default="reduced.{}.json".format(time.time()),
              help="File to store the reduced feature list and sample in")
@click.option("--n_features", default=60, type=int,
              help="Number of features to select")
@click.pass_context
def pca(ctx, output_file: str, n_features: int):
    """Principal component analysis that produces a file that can be used by the classifier commands"""
    X, y = ctx.obj["samples_and_features"]
    for i, x in enumerate(X):
        if len(x) == 0:
            print(i)
    X = preprocessing.normalize(X)
    ipca = IncrementalPCA(n_components=n_features)
    X = ipca.fit_transform(X, y)
    print("Number of selected features {}".format(ipca.n_components_))
    print("Components with maximum variance {}".format(ipca.components_))
    store_samples_and_features(X, y, output_file)


@learn.group()
@click.option("--file", help="Reduced data file")
@click.option("--verbose/--not_verbose", default=False, help="Should the classification process output information?")
@click.option("--runs", default=10, type=int,
              help="Cross validation runs")
@click.option("--n_jobs", default=-1, type=int,
              help="The number of parallel jobs to run for neighbors search. If -1, then the number of jobs is set to the number of CPU cores.")
@click.pass_context
def classify(ctx, file, verbose, runs, n_jobs):
    """Cross validate classifiers. The classifiers just call the corresponding scikit-learn classes."""
    X, y = load_samples_and_features(file)
    ctx.obj["sample_vector_size"] = len(X[0])
    ctx.obj["feature_num"] = len(set(y))
    ctx.obj["cross_val"] = lambda clf: cross_val(X, y, clf, file, runs, n_jobs)
    ctx.obj["cross_val_single_job"] = lambda clf: cross_val(X, y, clf, file, runs, 1)
    ctx.obj["verbose"] = verbose


@classify.command()
@click.option("--kernel", default="linear", type=click.Choice(["linear", "poly", "rbf", "sigmoid"]),
              help="Specifies the kernel type to be used in the algorithm")
@click.option("--probability/--no_probability", default=False,
              help="Whether to enable probability estimates. This must be enabled prior to calling fit, and will slow down that method.")
@click.option("--poly_degree", type=int, default=3,
              help="Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.")
@click.pass_context
def svc(ctx, kernel: str, probability: bool, poly_degree: int):
    """A wrapper around the support vector machine implementation of scikit-learn"""
    ctx.obj["cross_val"](SVC(kernel=kernel, verbose=ctx.obj["verbose"], probability=probability, degree=poly_degree))


@classify.command()
@click.option("--k", type=int, default=5,
              help="Number of considered neighbours")
@click.option("--algo", default="auto", type=click.Choice(["auto", "ball_tree", "kd_tree", "brute"]),
              help="""Algorithm used to compute the nearest neighbors:

    ‘ball_tree’ will use BallTree
    ‘kd_tree’ will use KDTree
    ‘brute’ will use a brute-force search.
    ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.""")
@click.option("--weights", type=click.Choice(["uniform", "distance"]), default="uniform",
              help="‘uniform’ : uniform weights. All points in each neighborhood are weighted equally. ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away. ")
@click.pass_context
def knn(ctx, k, algo, weights, n_jobs):
    """k-nearest-neighbor classifier"""
    ctx.obj["cross_val"](KNeighborsClassifier(n_neighbors=k, weights=weights,
                                              algorithm=algo, n_jobs=n_jobs))


@classify.command()
@click.option("--activation", default="relu", type=click.Choice(["identity", "logistic", "tanh", "relu"]),
              help="""

    Activation function for the hidden layer.

        ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
        ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
        ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
        ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)

""")
@click.option("--solver", default="adam", type=click.Choice(["lbfgs", "sgd", "adam"]),
              help="""
              The solver for weight optimization.

    ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
    ‘sgd’ refers to stochastic gradient descent.
    ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba

              """)
@click.option("--learning_rate", default="constant", type=click.Choice(["constant", "invscaling", "adaptive"]),
              help="""


    Learning rate schedule for weight updates.

        ‘constant’ is a constant learning rate given by ‘learning_rate_init’.
        ‘invscaling’ gradually decreases the learning rate learning_rate_ at each time step ‘t’ using an inverse scaling exponent of ‘power_t’. effective_learning_rate = learning_rate_init / pow(t, power_t)
        ‘adaptive’ keeps the learning rate constant to ‘learning_rate_init’ as long as training loss keeps decreasing. Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation score by at least tol if ‘early_stopping’ is on, the current learning rate is divided by 5.

    Only used when solver='sgd'.

              """)
@click.option("--max_iter", default=200, type=int,
              help="Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations.")
@click.pass_context
def mlp(ctx, **kwargs):
    """Multi-layer Perceptron classifier. Seems to be the most promising"""
    ctx.obj["cross_val"](MLPClassifier(verbose=ctx.obj["verbose"], **kwargs))


@classify.command()
@click.option("--dual/--primal", default=False,
              help="""Select the algorithm to either solve the dual or primal optimization problem. Prefer dual=False when n_samples > n_features.""")
@click.option("--max_iter", default=1000, type=int,
              help="The maximum number of iterations to be run.")
@click.pass_context
def linear_svc(ctx, **kwargs):
    """Linear Support Vector Classifier"""
    ctx.obj["cross_val"](LinearSVC(verbose=ctx.obj["verbose"], **kwargs))


@classify.command()
@click.option("--n_estimators", default=10, type=int,
              help="The number of trees in the forest.")
@click.pass_context
def random_forest(ctx, **kwargs):
    """A random forest classifier."""
    ctx.obj["cross_val"](RandomForestClassifier(verbose=ctx.obj["verbose"], **kwargs))



@classify.command()
@click.pass_context
def keras_mlp(ctx):
    """
    Multilayer Perceptron (MLP) for multi-class softmax classification.
    Based upon an keras example (uses tensorflow internally).
    """
    # based upon https://keras.io/getting-started/sequential-model-guide/
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.optimizers import SGD

    # Generate dummy data
    import numpy as np
    def build_model():
        model = Sequential()
        # Dense(64) is a fully-connected layer with 64 hidden units.
        # in the first layer, you must specify the expected input data shape:
        # here, 20-dimensional vectors.
        model.add(Dense(128, activation='relu', input_dim=ctx.obj["sample_vector_size"]))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(ctx.obj["feature_num"], activation='softmax'))

        sgd = SGD(lr=0.15, decay=1e-6, momentum=0.9, nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        return model

    from keras.wrappers.scikit_learn import KerasClassifier
    ctx.obj["cross_val_single_job"](KerasClassifier(build_model))


def cross_val(X, y, classificator, output_file, runs, n_jobs):
    scores = cross_val_score(classificator, np.array(X), y, cv=runs, n_jobs=n_jobs)
    print(scores)
    print("Accuracy: {:0.2f} (+/- {:0.2f})".format(scores.mean(), scores.std() * 2))

if __name__ == '__main__':
    cli()