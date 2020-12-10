# =======================================================================
# MULTICOLLINEARITY VIF FUNCTION
# (regression model assumptions test)
# =======================================================================
import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# =======================================================================
# HOMOSCEDASTICITY TEST PLOT FUNCTION
# =======================================================================
def plot_sced_test(df, target, features):
    """
    Utility for checking homoscedasticity assumption for regression models

    :param df: a pandas.DataFrame that contains the feature and target vars
    :param target: string representing the target var in the df
    :param features: a list of strings representing the feature vars in the df
    :return: nothing is returned the plots for viewing homoscedasticity are printed
    """

    for var in features:
        X = df[var]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=28
        )

        # Add a constant for the intercept
        # This will work but throw a warning.. to silence the warning you can pass X.values instead
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_train)

        lm_results = sm.OLS(y_train, X_train_const).fit()
        y_pred = lm_results.predict(X_test_const)

        plt.scatter(X_train, y_train, alpha=0.1)
        plt.plot(X_train, y_pred, c="red")
        plt.xlabel(var)
        plt.ylabel("Temperature")
        plt.show()

        lm_results.summary()

        true_residuals = lm_results.resid
        rand_norm_residuals = np.random.normal(0, 3, len(y_train))

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].scatter(X_train, rand_norm_residuals, alpha=0.2)
        axes[0].axhline(0, c="red", alpha=0.5)
        axes[0].set_xlabel(var)
        axes[0].set_ylabel("Residual")
        axes[0].set_title("What we want to see")

        axes[1].scatter(X_train, true_residuals, alpha=0.2)
        axes[1].axhline(0, c="red", alpha=0.5)
        axes[1].set_xlabel(var)
        axes[1].set_ylabel("Residual")
        axes[1].set_title("What we actually see")
        plt.show()


# =======================================================================GET
# TOP CORRELATIONS VS TARGET VAR & PLOT
# (data exploration)
# =======================================================================
def top_corrs(df, column, n=10):
    corr_df = df.corr()[[column]]
    corr_df.columns = ["corr"]
    corr_df["abs_corr"] = corr_df.abs()

    top_n_num = corr_df.sort_values("abs_corr", ascending=False).head(n)
    return top_n_num


def plot_top_corrs(df, column, n=10):
    top_corrs_df = top_corrs(df, column, n)
    top_feats = top_corrs_df.index
    top_corr = df[top_feats]
    top_corr_tall = pd.melt(top_corr, column)

    fg = sns.FacetGrid(top_corr_tall, col="variable", col_wrap=5, sharex=False)
    fg.map(sns.scatterplot, "value", column)
    plt.show()


# =======================================================================
# LINEAR REGRESSION PERFORMANCE DF
# =======================================================================
# pred_df = X_test.copy()
# pred_df["y_true"] = y_test
# pred_df["y_pred"] = y_pred
# pred_df["resid"] = pred_df["y_pred"] - pred_df["y_true"]
# pred_df["abs_resid"] = pred_df["resid"].abs()
# pred_df = pred_df.sort_values("abs_resid", ascending=False)
# pred_df.head()
# TODO find linear regression performance df info and convert to function

# =======================================================================,
# TUKEY'S METHOD - OUTLIERS
# (EDA)
# =======================================================================,

def tukey_outliers(df, var):
    """
    NumPy's `percentile()` method returns the
    values of the given percentiles. In this case,
    give `75` and `25` as parameters, which corresponds
    to the third and the first quartiles.
    """
    q75, q25 = np.percentile(df['Video views'], [75, 25])
    iqr = q75 - q25

    for threshold in np.arange(1, 5, 0.5):
        min_val = q25 - (iqr * threshold)
        max_val = q75 + (iqr * threshold)
        print('The score threshold is: {}'.format(threshold))
        print('Number of outliers is: {}'.format(len((np.where((df[var] > max_val) | (df[var] < min_val))[0]))))


# =======================================================================
# MULTICOLLINEARITY TEST - VARIANCE INFLATION FACTOR
# (model performance)
# =======================================================================

def print_vif(x):
    """
    Utility for checking multicollinearity assumption
    :param x: input features to check using VIF. This is assumed to be a pandas.DataFrame
    :return: nothing is returned the VIFs are printed as a pandas series
    """
    # Silence numpy FutureWarning about .ptp
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        x = sm.add_constant(x)

    vifs = []
    for i in range(x.shape[1]):
        vif = variance_inflation_factor(x.values, i)
        vifs.append(vif)

    print('VIF results\n-------------------------------')
    print(pd.Series(vifs, index=x.columns))
    print('-------------------------------\n')

# =======================================================================
# y = x
# fig = px.scatter(data_frame=pred_df, x="y_true", y="y_pred")
# fig.add_shape(type="line", x0=0, y0=0, x1=20, y1=20)
#
# fig = px.scatter(data_frame=pred_df, x="y_true", y="y_pred")
# fig.add_shape(
#     type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max()
# )
# =======================================================================

def predictions_df(X_test, y_test, y_preds):
    """
    Function to create a predictions dataframe from X_test, y_test, y_predictions input

    :param X_test:
    :param y_test:
    :param y_preds: X_test predictions; model.predict(X_test)
    """

    pred_df = X_test.copy()
    pred_df["y_true"] = y_test
    pred_df["y_preds"] = y_preds
    pred_df["residuals"] = pred_df.y_preds - pred_df.y_true
    pred_df["abs_residuals"] = pred_df.residuals.abs()
    pred_df = pred_df.sort_values("abs_residuals", ascending=True)

    return pred_df