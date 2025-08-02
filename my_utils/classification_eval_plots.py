import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


class ClassificationEvaluationPlots:
    @staticmethod
    def plot_precision_recall_curve(y_test, prediction_probabilities, show_baseline=True, ax=None):
        '''
            Plot a Precision-Recall curve for binary classification evaluation.
            
            The Precision-Recall curve is particularly useful for evaluating classifiers
            on imbalanced datasets where positive and negative classes are not evenly 
            distributed. It shows the trade-off between precision and recall across 
            different classification thresholds.
            
            Parameters
            ----------
            y_test : array-like of shape (n_samples,)
                True binary labels (0 or 1).
            prediction_probabilities : array-like of shape (n_samples,)
                Predicted probabilities for the positive class, typically from 
                model.predict_proba()[:, 1] or decision_function().
            show_baseline : bool, default=True
                Whether to display the no-skill baseline (random classifier performance).
                The baseline is a horizontal line at y = proportion of positive samples.
            ax : matplotlib.axes.Axes, optional
                Matplotlib axes object to plot on. If None, uses current axes.
            
            Notes
            -----
            - Most real world datasets are imblanced. So this is more useful than ROC curve
            - Area Under Curve (AUC-PR) is displayed as Average Precision (AP) score
            - Higher AP scores indicate better model performance
            - For imbalanced datasets, PR curves are more informative than ROC curves
            - Perfect classifier: AP = 1.0
            - Random classifier: AP ≈ proportion of positive samples
        '''

        if ax is None:
            ax = plt.gca()

        precisions, recalls, _ = precision_recall_curve(y_test, prediction_probabilities)
        
        # the area under curve. larger is better
        average_precision = average_precision_score(y_test, prediction_probabilities)

        ax.plot(recalls, precisions, label=f'Model: (AP = {average_precision:.2f})')
        
        # a random classifier without any skill will have an average prediction that's equal to the proportion of positive samples
        # the plot will be a straight line at the value of proportion of positive samples
        # if the curve is above it, then the model is predicting better than just guessing at random
        # if the curve is below it, then the model is predicting worse than just guessing at random
        if show_baseline:
            proportion_of_positive_samples  = len(y_test[y_test == 1]) / len(y_test)
            ax.axhline(y=proportion_of_positive_samples , alpha=0.7, color='black', linestyle='--',
                       label=f'Baseline (AP = {proportion_of_positive_samples :.2f})')

        # labels
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve')

        # Set axis limits for consistency
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        ax.legend()
        ax.grid(alpha=0.3)
    
    @staticmethod
    def plot_roc_curve(true_values, prediction_probabilities, show_baseline=True, ax=None):
        '''
            Plot a ROC (Receiver Operating Characteristic) curve for binary classification.
            
            The ROC curve shows the trade-off between True Positive Rate (sensitivity) 
            and False Positive Rate (1-specificity) across different classification 
            thresholds. It's useful for evaluating overall discriminative ability.
            
            Parameters
            ----------
            true_values : array-like of shape (n_samples,)
                True binary labels (0 or 1).
            prediction_probabilities : array-like of shape (n_samples,)
                Predicted probabilities for the positive class, typically from 
                model.predict_proba()[:, 1] or decision_function().
            show_baseline : bool, default=True
                Whether to display the no-skill baseline (random classifier).
                Shows as diagonal line with AUC = 0.5.
            ax : matplotlib.axes.Axes, optional
                Matplotlib axes object to plot on. If None, uses current axes.
            
            Notes
            -----
            - Area Under Curve (AUC-ROC) ranges from 0 to 1 (higher is better)
            - AUC = 1.0: Perfect classifier
            - AUC = 0.5: Random classifier (no discriminative ability)
            - AUC < 0.5: Worse than random (but predictions can be inverted)
                If your model consistently gets things wrong, it's actually learned something useful - just the opposite of what you wanted!
                Meaning, it has actually learned the true pattern but applied it backwards.
            - ROC curves can be overly optimistic on highly imbalanced datasets
        '''

        if ax is None:
            ax = plt.gca()
            
        false_positive_rate, true_positive_rate, _ = roc_curve(true_values, prediction_probabilities)
        # the area under curve. larger is better
        auc_value = auc(false_positive_rate, true_positive_rate)
        
        ax.plot(false_positive_rate, true_positive_rate, label=f'Model (AUC = {auc_value:.2f})')

        # a random classifier without any skill will have an AUC of 0.5, shown as diagonal line plot
        # if the curve is above it, then the model is predicting better than just guessing at random
        # if the curve is below it, then the model is predicting worse than just guessing at random
        if show_baseline:
            ax.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.7,
                    label='Baseline (AUC = 0.5)')
        
        # labels
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC curve')

        # Set axis limits for consistency
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        ax.legend()
        ax.grid(alpha=0.3)