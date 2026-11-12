import numpy as np
import pytest
import scipy.sparse as sp

from src.data.preprocessing import (
    GeneVocabulary, ensure_obs_columns, is_raw_counts, normalize_counts, select_hvg,
)


def test_detects_raw_counts_vs_normalized():
    X = sp.csr_matrix(np.random.default_rng(0).poisson(3, (30, 20)).astype(float))
    assert is_raw_counts(X)
    Xn, _ = normalize_counts(X)
    assert not is_raw_counts(Xn)


def test_normalize_gives_constant_depth_before_log():
    X = sp.csr_matrix(np.random.default_rng(0).poisson(3, (30, 20)).astype(float))
    Xn, lib = normalize_counts(X, log1p=False)
    tot = np.asarray(Xn.sum(1)).ravel()
    assert np.allclose(tot, 1e4, rtol=1e-4)
    assert np.allclose(lib, np.asarray(X.sum(1)).ravel())


def test_hvg_returns_requested_count():
    X = np.random.default_rng(0).poisson(2, (80, 50)).astype(np.float32)
    assert select_hvg(X, n_top=12).size == 12


def test_hvg_batch_aware_runs_and_is_sorted():
    rng = np.random.default_rng(0)
    X = rng.poisson(2, (80, 40)).astype(np.float32)
    b = rng.choice(["a", "b"], 80)
    idx = select_hvg(X, n_top=10, batch=b)
    assert idx.size == 10 and np.all(np.diff(idx) > 0)


def test_vocabulary_reserves_special_tokens():
    v = GeneVocabulary(["a", "b", "c"])
    assert len(v) == 6 and v.ids(["a"])[0] == 3
    assert v.ids(["zzz"])[0] == -1


def test_vocabulary_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        GeneVocabulary(["a", "a"])


def test_ensure_obs_records_what_it_fabricated():
    import anndata as ad
    import pandas as pd
    a = ad.AnnData(X=np.zeros((4, 3), dtype=np.float32),
                   obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]))
    ensure_obs_columns(a)
    assert "donor_id" in a.obs
    assert "donor_id" in a.uns["fabricated_obs"]
