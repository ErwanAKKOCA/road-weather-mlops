import numpy as np

from app.predictor import EMASequenceStore


def test_ema_is_causal_and_sequence_local():
    store = EMASequenceStore(alpha=0.6, ttl_seconds=3600, max_sequences=10)
    first, first_index = store.update("route-a", np.array([0.8, 0.1, 0.1]))
    second, second_index = store.update("route-a", np.array([0.2, 0.7, 0.1]))
    other, other_index = store.update("route-b", np.array([0.1, 0.2, 0.7]))

    assert first_index == 0
    assert second_index == 1
    assert other_index == 0
    np.testing.assert_allclose(first, [0.8, 0.1, 0.1])
    np.testing.assert_allclose(second, 0.6 * np.array([0.2, 0.7, 0.1]) + 0.4 * first)
    np.testing.assert_allclose(other, [0.1, 0.2, 0.7])


def test_reset_restarts_frame_index():
    store = EMASequenceStore(alpha=0.6, ttl_seconds=3600, max_sequences=10)
    store.update("route", np.array([1.0, 0.0, 0.0]))
    reset_values, index = store.update("route", np.array([0.0, 1.0, 0.0]), reset=True)
    assert index == 0
    np.testing.assert_allclose(reset_values, [0.0, 1.0, 0.0])
