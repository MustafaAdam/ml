import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize


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



    def plot_precision_recall_curve_multiclass(y_true, y_probs, classes, 
                                            show_baseline=True, 
                                            average="macro", 
                                            show_only_average=False, 
                                            ax=None):
        """
        Plot Precision-Recall curves for multiclass classification.

        Parameters
        ----------
        y_true : array of shape (n_samples,)
            True class labels.
        y_probs : array of shape (n_samples, n_classes)
            Predicted probabilities for each class.
        classes : array-like
            Class labels.
        show_baseline : bool, default=True
            Show no-skill baseline (positive class proportion).
        average : {"micro", "macro"}, default="macro"
            Averaging method for summary PR curve.
        average_only : bool, default=False
            If True, plot only the averaged curve (no per-class curves).
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        """
        if ax is None:
            ax = plt.gca()

        y_bin = label_binarize(y_true, classes=classes)

        if not show_only_average:
            for i, class_label in enumerate(classes):
                precision, recall, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
                ap = average_precision_score(y_bin[:, i], y_probs[:, i])
                ax.plot(recall, precision, lw=1.5,
                        label=f'Class {class_label} (AP={ap:.2f})')

        if average == "micro":
            precision, recall, _ = precision_recall_curve(y_bin.ravel(), y_probs.ravel())
            ap = average_precision_score(y_bin, y_probs, average="micro")
            ax.plot(recall, precision, color='navy', linestyle='--', lw=3,
                    label=f'Micro-average (AP={ap:.2f})')
        elif average == "macro":
            ap = average_precision_score(y_bin, y_probs, average="macro")
            precision, recall, _ = precision_recall_curve(y_bin.ravel(), y_probs.ravel())
            ax.plot(recall, precision, color='darkorange', linestyle='--', lw=3,
                    label=f'Macro-average (AP={ap:.2f})')

        if show_baseline:
            baseline = np.mean(y_true == classes[0])
            ax.axhline(y=baseline, color='black', linestyle='--', alpha=0.7,
                    label=f'Baseline (AP={baseline:.2f})')

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)


    def plot_roc_curve_multiclass(y_true, y_probs, classes, show_baseline=True, average="macro", show_only_average=False, ax=None):
        """
        Plot ROC curves for multiclass classification.

        Parameters
        ----------
            y_true : array of shape (n_samples,)
                True class labels.
            y_probs : array of shape (n_samples, n_classes)
                Predicted probabilities for each class.
            classes : array-like
                Class labels.
            show_baseline : bool, default=True
                Show no-skill baseline.
            average : {"micro", "macro"}, default="macro"
                Averaging method for summary ROC curve.
            average_only : bool, default=False
                If True, plot only the average curve (no per-class curves).
            ax : matplotlib.axes.Axes, optional
                Axes to plot on.
        """
        if ax is None:
            ax = plt.gca()

        y_bin = label_binarize(y_true, classes=classes)

        if not show_only_average:
            # Per-class curves
            for i, class_label in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=1.5, label=f'Class {class_label} (AUC={roc_auc:.2f})')

        # Average curve
        if average == "micro":
            fpr, tpr, _ = roc_curve(y_bin.ravel(), y_probs.ravel())
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color='navy', linestyle='--', lw=3,
                    label=f'Micro-average (AUC={roc_auc:.2f})')
        elif average == "macro":
            all_fpr = np.unique(np.concatenate(
                [roc_curve(y_bin[:, i], y_probs[:, i])[0] for i in range(len(classes))]
            ))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(len(classes)):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
                mean_tpr += np.interp(all_fpr, fpr, tpr)
            mean_tpr /= len(classes)
            roc_auc = auc(all_fpr, mean_tpr)
            ax.plot(all_fpr, mean_tpr, color='darkorange', linestyle='--', lw=3,
                    label=f'Macro-average (AUC={roc_auc:.2f})')

        if show_baseline:
            ax.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.7,
                    label='Baseline (AUC=0.5)')

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

