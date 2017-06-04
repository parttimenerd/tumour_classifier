"""
CLI part of the project that also wraps the calls to scikit-learn.

Some of the help texts are copied from the scikit-learn documentation
"""
from pprint import pprint

import numpy as np
from collections import defaultdict

from sklearn import preprocessing
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import scipy as sp

import sys
import json
import os

import click
import time

from typing import List, Dict, Tuple, Set

from tabulate import tabulate

from preprocessing import DB, knn_impute, MI_RNA_NAMES, Sample


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
@click.option("--discard_missing/--no_discards", default=True,
              help="Discard miRNAs that are missing for more than 20% of the samples")
@click.pass_context
def learn(ctx, blood: bool, only_blood: bool, min_samples: int, discard_missing: bool):
    """Group of commands that work on the database"""
    ctx.obj["blood"] = blood
    ctx.obj["only_blood"] = only_blood
    X, y = ctx.obj["db"].sample_arrs_and_features(blood_normals=blood, tumour=not only_blood,
                                                                             min_samples=min_samples)
    if discard_missing:
        X = _discard_missing(X)
    ctx.obj["samples_and_features"] = X, y
    ctx.obj["min_samples"] = min_samples


def _discard_missing(X: List[List[float]], min_nz_samples: float = 0.8) -> List[List[float]]:
    count = [0 for i in range(len(X[0]))]
    for x in X:
        for i in range(len(x)):
            if x[i] is not 0:
                count[i] += 1
    indexes = [i for i in range(len(count)) if count[i] < 0.8 * len(X)]
    #print("discarded {}".format(len(indexes)))
    return _rm_indexes(X, set(indexes))

def _rm_indexes(X: List[List[float]], indexes: Set[int]) -> List[List[float]]:
    ret = []
    for x in X:
        ret.append([x[i] for i in range(len(x)) if i not in indexes])
    return ret

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
@click.option("--rfe/--no_rfe", default=False,
              help="Use recursive feature elimination")
@click.option("--rfe_n_features", type=int, default=60,
              help="Number of features to select")
@click.option("--rfe_step", type = int, default=10,
              help="Step size for feature elimination")
@click.option("--rfe_verbose/--rfe_silent", default=False,
              help="Should the feature elimination print details")
@click.option("--param_opt/--no_param_opt", default=False, help="Do a parameter optimization")
@click.option("--param_opt_iters", default=60, type=int,
              help="Parameter optimization iterations")
@click.pass_context
def classify(ctx, file, verbose, runs, n_jobs, rfe: bool, rfe_n_features: int, rfe_step, rfe_verbose: bool,
             param_opt: bool, param_opt_iters: bool):
    """Cross validate classifiers. The classifiers just call the corresponding scikit-learn classes."""
    X, y = None, None
    use_rfe = False
    try: # reduced file
        X, y = load_samples_and_features(file)
    except: # database file
        X, y = ctx.obj["samples_and_features"]
    ctx.obj["sample_vector_size"] = len(X[0])
    ctx.obj["feature_num"] = len(set(y))
    args = {
        "X": X, "y": y, "output_file": file, "runs": runs, "n_jobs": n_jobs,
        "rfe": rfe, "rfe_n_features": rfe_n_features, "rfe_step": rfe_step,
        "rfe_verbose": rfe_verbose, "param_opt": param_opt, "param_opt_iters": param_opt_iters,
        "param_opt_verbose": 2 if verbose else 0
    }
    single_job_args = dict(args)
    args["n_jobs"] = 1
    ctx.obj["cross_val"] = lambda clf, dist: cross_val(**args, classificator=clf, param_opt_dist=dist)
    ctx.obj["cross_val_single_job"] = lambda clf, dist: cross_val(**single_job_args, classificator=clf, param_opt_dist=dist)
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
    ctx.obj["cross_val"](SVC(kernel=kernel, verbose=ctx.obj["verbose"], probability=probability, degree=poly_degree),
                         {'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                          'C': [1, 10, 100, 1000]}
                         )

@learn.command()
@click.option("--cv", default=10, type=int, help="Number of cross validations")
@click.option("--dump_filename", default="trials.dump")
@click.option("--dump_period", type=int, default=3)
@click.pass_context
def param_opt(ctx, cv: int, dump_filename: str, dump_period: int):
    """
    Estimate the best preprocessor and classifier. Outputs its results continuously.
    """
    from hpsklearn import HyperoptEstimator
    import hpsklearn
    from hyperopt import tpe
    import pickle

    estimator = HyperoptEstimator(preprocessing=hpsklearn.components.any_preprocessing('pp'),
                              algo=tpe.suggest)
    X, y = ctx.obj["samples_and_features"]
    #estimator.fit(X, y)
    fit_iterator = estimator.fit_iter(X, y)
    next(fit_iterator)
    cur_loss = 1
    while True:
        fit_iterator.send(1)
        try:
            loss = estimator.trials.best_trial["result"]["loss"]
            if loss < cur_loss:
                print("Ran {} trials".format(len(estimator.trials)))
                print(estimator.best_model())
                pprint(estimator.trials.best_trial["result"])
                cur_loss = loss
            if loss == cur_loss or len(estimator.trials) % dump_period:
                if dump_filename is not None:
                    with open(dump_filename, 'wb') as dump_file:
                        pickle.dump(estimator.trials, dump_file)

            sys.stdout.write("+")
            sys.stdout.flush()
        except Exception as ex:
            print(ex)
            pass


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
def knn(ctx, k, algo, weights):
    """k-nearest-neighbor classifier"""
    ctx.obj["cross_val"](KNeighborsClassifier(n_neighbors=k, weights=weights,
                                              algorithm=algo),
                         {"weights": ["uniform", "distance"],
                          "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                          "p": [2,3,4,5],
                          "n_neighbors": [5,6,7,8,9,10]}
                         )


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
@click.option("--max_iter", default=500, type=int,
              help="Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations.")
@click.pass_context
def mlp(ctx, **kwargs):
    """Multi-layer Perceptron classifier. Seems to be the most promising"""
    ctx.obj["cross_val"](MLPClassifier(verbose=ctx.obj["verbose"], **kwargs), {
        "activation": ["identity", "logistic", "tanh", "relu"],
        "solver": ["lbfgs", "sgd", "adam"],
        "alpha": [0.001, 0.005, 0.1, 0.0001, 0.0005, 0.00001],
        "learning_rate": ["constant", "invscaling", "adaptive"]
    })


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


def cross_val(X, y, classificator, output_file, runs: int, n_jobs: int, rfe: bool, rfe_n_features: int, rfe_step: int,
              rfe_verbose: bool, param_opt: bool, param_opt_dist: dict, param_opt_iters: int,
              param_opt_verbose: bool):
    if rfe:
        selector = RFE(classificator, verbose=rfe_verbose, step=rfe_step, n_features_to_select=rfe_n_features)
        X = selector.fit_transform(X, y)
    if param_opt:
        random_search = RandomizedSearchCV(classificator, param_distributions=param_opt_dist, n_iter=param_opt_iters,
                                           verbose=param_opt_verbose, n_jobs=-1)
        start = time.time()
        random_search.fit(X, y)
        print("RandomizedSearchCV took {:.2f} seconds for {:d} candidates"
              " parameter settings.".format((time.time() - start), param_opt_iters))
        _report(random_search.cv_results_)
    else:
        scores = cross_val_score(classificator, np.array(X), y, cv=runs, n_jobs=n_jobs)
        print(scores)
        print("Accuracy: {:0.2f} (+/- {:0.2f})".format(scores.mean(), scores.std() * 2))



# Utility function to report best scores
# from http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html#sphx-glr-auto-examples-model-selection-randomized-search-py
def _report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


@cli.group()
def misc():
    pass

@misc.command()
@click.option("--per_type/--all", default=False, help="Group the miRNAs by type")
@click.option("--asc/--desc", default=True)
@click.option("--limit", default=10, type=int, help="Only show n entries per table, -1 for all entries")
@click.pass_context
def mirna_variance(ctx, per_type: bool, asc: bool, limit: int):
    from tabulate import tabulate
    """
    Calculate the variance of the expression levels of all miRNAs
    and output it
    """
    def table(samples: List[Sample]):
        rows = []
        miRNA_values = _mirna_value(samples)
        for miRNA, values in miRNA_values.items():
            rows.append([miRNA, sp.std(values), sp.mean(values)])
        rows = sorted(rows, key=lambda e: e[1])
        if not asc:
            rows = list(reversed(rows))
        if limit is not -1:
            rows = rows[0:limit]
        print(tabulate(rows, headers=["miRNA", "std", "mean"]))

    db = ctx.obj["db"]  # type: DB

    if per_type:
        samples_per_type = db.samples_per_type()
        for type in sorted(samples_per_type.keys()):
            print(type)
            table(samples_per_type[type])
    else:
        table(db.samples.values())


@misc.command()
@click.option("--asc/--desc", default=True)
@click.option("--limit", default=10, type=int, help="Only show n entries per table, -1 for all entries")
@click.pass_context
def mirna_corr(ctx, asc: bool, limit: int):
    db = ctx.obj["db"]  # type: DB
    types = {}

    l = []  # type: List[Tuple[str, List[float], List[int]]]
    for sample in db.samples.values():
        if not l:
            l = [(n, [], []) for n in sample.mi_rna_profile.mi_rna_names]
        if sample.tissue_type not in types:
            types[sample.tissue_type] = len(types)
        type = types[sample.tissue_type]
        arr = sample.mi_rna_profile.reads_per_million
        for i in range(len(arr)):
            l[i][1].append(arr[i])
            l[i][2].append(type)
    rows = []  # type: List[Tuple[str, float]]  # miRNA, correlation
    for miRNA, vals, types in l:
        rows.append((miRNA, sp.stats.pearsonr(vals, types)))
    if not asc:
        rows = sorted(rows, key=lambda e: e[1])
    if not asc:
        rows = list(reversed(rows))
    if limit is not -1:
        rows = rows[0:limit]
    print(tabulate(rows, headers=["miRNA", "pearsonr"]))


def _mirna_value(samples: List[Sample]) -> Dict[str, List[float]]:
    d = []
    names = list(samples)[0].mi_rna_profile.mi_rna_names
    d = [[] for n in names]
    for sample in samples:
        reads = sample.mi_rna_profile.reads_per_million
        for i in range(len(reads)):
            d[i].append(reads[i])
    ret = {names[i]:d[i] for i in range(len(d))}
    return ret


if __name__ == '__main__':
    cli()