"""Test suite for Client-Side Orthogonal Privacy Projector."""

from __future__ import annotations

import unittest
import numpy as np

from apical.orthogonal import OrthogonalPrivacyProjector


class TestOrthogonalPrivacyProjector(unittest.TestCase):
    """Verify isometric properties of the keyed client-side privacy projector."""

    def test_orthogonality_and_dimension(self) -> None:
        d = 64
        projector = OrthogonalPrivacyProjector(privacy_secret="secret_key_1", in_features=d)
        _u = projector.matrix

        self.assertEqual(_u.shape, (d, d))
        # Operator transpose product must equal identity within float32 tolerance
        eye = np.eye(d, dtype=np.float32)
        diff = np.max(np.abs(_u.T @ _u - eye))
        self.assertLess(diff, 1e-4, f"Operator is not isometric: max diff = {diff}")

    def test_norm_preservation(self) -> None:
        d = 128
        projector = OrthogonalPrivacyProjector(privacy_secret="test_norm", in_features=d)

        # Random vector r
        r = np.random.randn(d).astype(np.float32)
        r_norm = float(np.linalg.norm(r))

        r_tilde = projector.project(r)
        r_tilde_norm = float(np.linalg.norm(r_tilde))

        self.assertAlmostEqual(r_norm, r_tilde_norm, places=4)

    def test_inner_product_preservation(self) -> None:
        d = 64
        projector = OrthogonalPrivacyProjector(privacy_secret="test_dot", in_features=d)

        r1 = np.random.randn(d).astype(np.float32)
        r2 = np.random.randn(d).astype(np.float32)

        dot_original = float(np.dot(r1, r2))

        r1_tilde = projector.project(r1)
        r2_tilde = projector.project(r2)

        dot_projected = float(np.dot(r1_tilde, r2_tilde))
        self.assertAlmostEqual(dot_original, dot_projected, places=4)

    def test_determinism_and_key_divergence(self) -> None:
        d = 32
        p1 = OrthogonalPrivacyProjector(privacy_secret="identical_secret", in_features=d)
        p2 = OrthogonalPrivacyProjector(privacy_secret="identical_secret", in_features=d)
        p3 = OrthogonalPrivacyProjector(privacy_secret="different_secret", in_features=d)

        # Same secret produces identical projection matrix
        np.testing.assert_allclose(p1.matrix, p2.matrix, atol=1e-6)

        # Different secret produces diverging matrix
        diff = np.max(np.abs(p1.matrix - p3.matrix))
        self.assertGreater(diff, 0.5)

    def test_batch_projection(self) -> None:
        d = 32
        batch_size = 10
        projector = OrthogonalPrivacyProjector(privacy_secret="batch_secret", in_features=d)

        batch = np.random.randn(batch_size, d).astype(np.float32)
        projected_batch = projector.project_batch(batch)

        self.assertEqual(projected_batch.shape, (batch_size, d))

        # Check each row matches individual projection
        for i in range(batch_size):
            single = projector.project(batch[i])
            np.testing.assert_allclose(projected_batch[i], single, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
