import numpy as np

from mopa.features import occupancy, standardize, window


def test_window_masks_future_steps():
    pos = np.arange(2 * 6 * 2, dtype=np.float32).reshape(2, 6, 2)

    out = window(pos, k=2, kmax=5).reshape(2, 5, 2)

    np.testing.assert_array_equal(out[:, :2], pos[:, :2])
    assert np.all(out[:, 2:] == 0.0)


def test_standardize_reuses_train_statistics():
    train = np.array([[1.0, 3.0], [3.0, 7.0]], dtype=np.float32)
    val = np.array([[5.0, 11.0]], dtype=np.float32)

    ztrain, mu, sd = standardize(train)
    zval, _, _ = standardize(val, mu=mu, sd=sd)

    np.testing.assert_allclose(ztrain.mean(0), 0.0, atol=1e-6)
    np.testing.assert_allclose(zval, (val - mu) / sd)


def test_occupancy_rows_are_normalized():
    pos = np.array(
        [
            [[-0.5, -0.5], [-0.5, -0.5], [0.5, 0.5]],
            [[0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]],
        ],
        dtype=np.float32,
    )

    occ = occupancy(pos, 0, 3, bins=4, value_range=(-1.0, 1.0))

    assert occ.shape == (2, 16)
    np.testing.assert_allclose(occ.sum(1), 1.0)
    assert np.count_nonzero(occ[0]) == 2


def test_occupancy_rejects_invalid_shape():
    bad = np.zeros((4, 10), dtype=np.float32)

    try:
        occupancy(bad, 0, 5)
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("occupancy should reject non-position tensors")
