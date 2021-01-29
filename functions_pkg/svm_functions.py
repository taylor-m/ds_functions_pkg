# =======================================================================
# SUPPORT VECTOR MACHINES CLASSIFIER
# (FIT, SCORE, PLOT)
# =======================================================================
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC

def svm_fit_score_plot(**kwargs):
    print(kwargs)

    model = SVC(**kwargs)
    model.fit(X_train, y_train)

    print(f"\nTrain score: {model.score(X_train, y_train)}")
    print(f"Test score: {model.score(X_test, y_test)}")

    plot_decision_regions(X_train, y_train, clf=model, scatter_kwargs={"alpha": 0.05})
    plt.show()