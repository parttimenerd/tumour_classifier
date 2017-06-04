MicroRNA classification
=======================

This project can pull microRNA expression profiles from [The Cancer Genome Atlas](https://cancergenome.nih.gov/), store it in a JSON file, preprocess and build
a classifier for tumour tissue from it. Different classifiers can be used.
For each classifier the cross validated accuracy is calculated.

Inspired by the paper ["MicroRNA based Pan-Cancer Diagnosis and Treatment Recommendation"](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-1421-y) by Cheerla and Gevaert and done (just for fun and profit) 
as part of the [Hot-Topics in Bioinformatics Seminar](https://sco.h-its.org/exelixis/web/teaching/BioinformaticsModule.html#seminar) at KIT.

Workflow
--------
(See the `--help` pages of the commands for more information on how to use them.)
1. Pull the data from the TCGA into a database (calling its API)
    - you can use the database from [here](http://mostlynerdless.de/files/women_db.zip) with the data of all
      female participants (1490 cases and 1612 samples)
    - or use `python3 classifier.py update_db [case id 1] […]`
    - or use `python3 classifier.py update_db_from_dict [file]`
    - → creates a database
2. Impute the missing normal tissue data → creates a new database
    - use `python3 classifier.py impute`
    - might fail for databases that are to large
    - and might not lead to better results
3. Reduce the feature space via a principal component analysis → reduced JSON
    - use `python3 classifier.py learn pca`
    - only the resulting file is usable for the next step
4. Create some classifiers and cross validate them
    - use `python3 classifier.py learn classify [svc, …]`
    
Example workflow: (assumes that the mentioned prepared database is used and stored in a file called `women_db`)

```
python3 classifier.py --db women_db impute women_db_imputed
python3 classifier.py --db women_db_imputed learn pca --output_file reduced_women_db
python3 classifier.py learn classify --file reduced_women_db mlp
```

Why?
----
Because it was a nice way to learn a bit machine learning and to better understand
a paper.

Dependencies
------------
- python3 (3.4 or later)
- scipy
- scikit-learn
- click
- keras and tensorflow (for `keras_mpl` classifier)
- hyperopt and hyperopt-sklearn for the parameter optimization

License
-------
MIT
