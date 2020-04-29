Software Dependencies
------------------------------
(pip3 install <library>)
- Python 3.7.2
- numpy 1.18.1
- scikit-learn (sklearn) 0.22.1 
- Keras 2.3.1
- Keras-Applications 1.0.8
- Keras-Preprocessing 1.1.0
- scipy 1.4.1
- tensorflow 2.1.0
- imbalanced-learn (imblearn) 0.6.2

Data set is included in .zip file under folder 'abalone'

Instructions
------------------------------
To start up program:
python3 fp.py

Running each algorithm will create the following outputs:
- precision, recall, f1-score table (classification_report)
- best hyperparameter found by Grid Search Cross Validation
- detail results of Grid Search Cross Validation
- hyperparameter tuning time
- training time
- algorithm performance determined by k-fold Cross Validation and weighted f1 score (mean, confidence interval)

To run Random Forest with Bagging:
uncomment lines 269 to 272

To run Support Vector Machines:
uncomment lines 275 to 278

To run Neural Network:
uncomment lines 281 to 284
