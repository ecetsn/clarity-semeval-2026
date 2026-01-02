"""
Evaluation utilities: metrics, results printing, tables, plots
"""

from .metrics import compute_all_metrics, print_classification_report
from .tables import (
    create_results_table,
    print_results_table,
    create_final_summary_pivot,
    create_model_wise_summary_pivot,
    create_classifier_wise_summary_pivot,
    style_table
)
from .plots import (
    plot_confusion_matrix,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_metrics_comparison
)
from .visualizer import visualize_all_evaluation, visualize_comparison

__all__ = [
    'compute_all_metrics',
    'print_classification_report',
    'create_results_table',
    'print_results_table',
    'create_final_summary_pivot',
    'create_model_wise_summary_pivot',
    'create_classifier_wise_summary_pivot',
    'style_table',
    'plot_confusion_matrix',
    'plot_precision_recall_curves',
    'plot_roc_curves',
    'plot_metrics_comparison',
    'visualize_all_evaluation',
    'visualize_comparison'
]

