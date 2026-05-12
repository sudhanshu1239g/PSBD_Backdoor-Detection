import numpy as np
import pytest

from psbd_nlp.scoring import mean_l1_shift, softmax, threshold_scores


def test_softmax_rows_sum_to_one():
    probs = softmax(np.array([[1.0, 2.0], [3.0, 1.0]]))

    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs[0, 1] > probs[0, 0]


def test_mean_l1_shift_scores_each_sample():
    baseline = np.array([[0.9, 0.1], [0.4, 0.6]])
    perturbed = np.array(
        [
            [[0.8, 0.2], [0.45, 0.55]],
            [[0.7, 0.3], [0.35, 0.65]],
        ]
    )

    scores = mean_l1_shift(baseline, perturbed)

    assert scores.shape == (2,)
    assert scores[0] > scores[1]


def test_threshold_scores_low_tail():
    flags, cutoff = threshold_scores(np.array([0.01, 0.2, 0.3, 0.4]), 0.25, "low")

    assert cutoff == pytest.approx(0.01)
    assert flags.tolist() == [True, False, False, False]
