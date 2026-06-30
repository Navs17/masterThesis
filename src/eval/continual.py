"""Continual learning evaluation metrics: BWT and Forgetting.

result_matrix[t][i] = accuracy (or F1) on task i after training through task t.
Rows are indexed by the most-recently-trained task; columns by the evaluated task.
"""


def compute_bwt(result_matrix):
    """Backward Transfer: mean change in performance on old tasks after all training.

    Negative BWT = forgetting; positive = backward knowledge transfer.
    BWT = (1 / T-1) * sum_{i=0}^{T-2} (R[T-1][i] - R[i][i])
    """
    T = len(result_matrix)
    if T <= 1:
        return 0.0
    return sum(result_matrix[T - 1][i] - result_matrix[i][i] for i in range(T - 1)) / (T - 1)


def compute_forgetting(result_matrix):
    """Average Forgetting: mean peak-to-final performance drop across old tasks.

    Forgetting on task i = R[i][i] - R[T-1][i]  (clamped to 0 if negative)
    """
    T = len(result_matrix)
    if T <= 1:
        return 0.0
    forgettings = [max(0.0, result_matrix[i][i] - result_matrix[T - 1][i]) for i in range(T - 1)]
    return sum(forgettings) / len(forgettings)
