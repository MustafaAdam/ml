import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

class EvaluationPlots:
    @staticmethod
    def plot_precision_recall_curve(y_test, prediction_probabilities, ax=None):
        '''
            The Precision-Recall Curve is a tool to evaluate a classifier—especially useful
            for imbalanced datasets where positive and negative classes are not evenly distributed.
        '''

        if ax is None:
            ax = plt.gca()

        precisions, recalls, _ = precision_recall_curve(y_test, prediction_probabilities)

        ax.plot(recalls, precisions)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.grid()
    
    @staticmethod
    def plot_roc_area_curve(true_values, prediction_probabilities, ax=None):
        '''
            Plot the ROC (Receiver Operating Characteristic) area curve.
        '''

        if ax is None:
            ax = plt.gca()
            
        false_positive_rate, true_positive_rate, _ = roc_curve(true_values, prediction_probabilities)
        auc_value = auc(false_positive_rate, true_positive_rate)
        
        ax.plot(false_positive_rate, true_positive_rate, label=f'Model (AUC = {auc_value:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
        
        ax.set_xlabel('False Positive Rates')
        ax.set_ylabel('True Positive Rates')
        ax.set_title('ROC area curve')
        ax.legend()
        ax.grid()