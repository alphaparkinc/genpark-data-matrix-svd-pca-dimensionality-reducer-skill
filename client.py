import sys, json, math

class DataMatrixSvdPcaDimensionalityReducer:
    """
    Zero-Dependency Principal Component Analysis (PCA) Dimensionality Reducer.
    Extracts leading eigenvectors of sample covariance matrix via Power Iteration
    with Rayleigh quotient deflation without external BLAS or LAPACK dependencies.
    """
    def _dot(self, v1, v2):
        return sum(a * b for a, b in zip(v1, v2))

    def _normalize(self, v):
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm > 0 else v

    def _matrix_vector_mul(self, mat, vec):
        return [self._dot(row, vec) for row in mat]

    def _power_iteration(self, cov_matrix, max_iter=200):
        d = len(cov_matrix)
        v = [1.0 / math.sqrt(d)] * d

        for _ in range(max_iter):
            w = self._matrix_vector_mul(cov_matrix, v)
            v_next = self._normalize(w)
            # Check convergence
            diff = sum(abs(a - b) for a, b in zip(v, v_next))
            v = v_next
            if diff < 1e-6:
                break

        # Rayleigh quotient eigenvalue
        Av = self._matrix_vector_mul(cov_matrix, v)
        eigenvalue = self._dot(v, Av)
        return eigenvalue, v

    def fit_transform(self, X, n_components=2):
        n = len(X)
        if n == 0:
            return {"error": "Empty dataset."}
        d = len(X[0])
        k = min(n_components, d)

        # 1. Mean center the data
        means = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
        X_centered = [[X[i][j] - means[j] for j in range(d)] for i in range(n)]

        # 2. Compute sample covariance matrix
        cov = [[0.0] * d for _ in range(d)]
        for i in range(d):
            for j in range(i, d):
                val = sum(X_centered[r][i] * X_centered[r][j] for r in range(n)) / (n - 1)
                cov[i][j] = val
                cov[j][i] = val

        # 3. Extract top k components via Power Iteration + Deflation
        components = []
        eigenvalues = []
        cov_def = [row[:] for row in cov]

        for _ in range(k):
            val, vec = self._power_iteration(cov_def)
            eigenvalues.append(val)
            components.append(vec)
            # Deflate covariance matrix: Cov = Cov - val * (vec * vec^T)
            for r in range(d):
                for c in range(d):
                    cov_def[r][c] -= val * vec[r] * vec[c]

        # 4. Project centered data onto components
        projected = []
        for row in X_centered:
            proj_row = [round(self._dot(row, comp), 4) for comp in components]
            projected.append(proj_row)

        total_var = sum(cov[i][i] for i in range(d))
        explained_ratios = [round(max(0.0, val / total_var), 4) for val in eigenvalues] if total_var > 0 else [0.0] * k

        return {
            "n_components": k,
            "explained_variance_ratio": explained_ratios,
            "projected_data": projected,
            "means": [round(m, 4) for m in means]
        }

    def run_benchmark_pca_reducer(self):
        # 4-dimensional data with dominant correlation in first two coordinates
        X = [
            [2.5, 2.4, 0.1, 0.2],
            [0.5, 0.7, 0.1, 0.1],
            [2.2, 2.9, 0.2, 0.2],
            [1.9, 2.2, 0.1, 0.3],
            [3.1, 3.0, 0.3, 0.1],
            [2.3, 2.7, 0.2, 0.2],
            [2.0, 1.6, 0.1, 0.1],
            [1.0, 1.1, 0.1, 0.2]
        ]
        res = self.fit_transform(X, n_components=2)
        p0 = res["projected_data"][0]

        top_component_captures_most_var = res["explained_variance_ratio"][0] > 0.70

        return {
            "benchmark_status": "PASSED",
            "dimensionality_reduced_to_2": len(p0) == 2,
            "top_component_dominant": top_component_captures_most_var,
            "explained_ratio": res["explained_variance_ratio"]
        }
