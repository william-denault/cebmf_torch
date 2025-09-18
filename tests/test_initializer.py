import torch

from cebmf_torch import cEBMF
from cebmf_torch.torch_main import NoiseType


def test_simple_init():
    """Test initialization with simple parameters."""
    Y = torch.randn(10, 5)

    # Test basic initialization
    model = cEBMF(data=Y, K=3, prior_L="norm", prior_F="laplace")

    assert model.model.K == 3
    assert model.model.prior_L == "norm"
    assert model.model.prior_F == "laplace"
    assert model.Y.shape == (10, 5)
    assert hasattr(model, "L")
    assert hasattr(model, "F")


def test_all_parameters():
    """Test initialization with all parameters."""
    Y = torch.randn(15, 8)
    X_l = torch.randn(15, 3)
    X_f = torch.randn(8, 2)

    L_kwargs = {"penalty": 2.0, "shuffle": True}
    F_kwargs = {"max_iter": 20, "tol": 1e-4}

    model = cEBMF(
        data=Y,
        K=5,
        prior_L="norm",
        prior_F="exp",
        allow_backfitting=False,
        prune_thresh=0.1,
        noise_type=NoiseType.ROW_WISE,
        X_l=X_l,
        X_f=X_f,
        prior_L_kwargs=L_kwargs,
        prior_F_kwargs=F_kwargs,
        self_row_cov=True,
        self_col_cov=True,
    )
    model.initialise_factors()

    # Check model params
    assert model.model.K == 5
    assert model.model.prior_L == "norm"
    assert model.model.prior_F == "exp"
    assert model.model.allow_backfitting is False
    assert model.model.prune_thresh == 0.1

    # Check noise params
    assert model.noise.type == NoiseType.ROW_WISE

    # Check covariate params
    assert torch.equal(model.covariate.X_l, X_l)
    assert torch.equal(model.covariate.X_f, X_f)
    assert model.covariate.self_row_cov is True
    assert model.covariate.self_col_cov is True

    # Check prior kwargs
    for key, value in L_kwargs.items():
        assert model.prior_L_fn.kwargs[key] == value
    for key, value in F_kwargs.items():
        assert model.prior_F_fn.kwargs[key] == value


def test_defaults():
    """Test that default parameters work correctly."""
    Y = torch.randn(10, 5)

    # Minimal initialization
    model = cEBMF(data=Y)

    # Check defaults
    assert model.model.K == 5  # Default K
    assert model.model.prior_L == "norm"
    assert model.model.prior_F == "norm"
    assert model.model.allow_backfitting is True
    assert model.model.prune_thresh == 0.999  # DEFAULT_PRUNE_THRESH

    # Check noise defaults
    assert model.noise.type == NoiseType.CONSTANT

    # Check covariate defaults
    assert model.covariate.X_l is None
    assert model.covariate.X_f is None
    assert model.covariate.self_row_cov is False
    assert model.covariate.self_col_cov is False


def test_device_handling():
    """Test device parameter handling."""
    Y = torch.randn(10, 5)

    # Test default device
    model1 = cEBMF(data=Y, K=3)
    assert model1.device.type in ["cpu", "cuda"]

    # Test explicit device
    device = torch.device("cpu")
    model2 = cEBMF(data=Y, K=3, device=device)
    assert model2.device == device
    assert model2.Y.device == device


def test_basic_functionality():
    """Test that the initialized model works correctly."""
    Y = torch.randn(20, 15)

    # Initialize model
    model = cEBMF(data=Y, K=3, prior_L="norm", prior_F="norm")

    # Test that basic functionality works
    model.initialise_factors()
    assert model.L.shape == (20, 3)
    assert model.F.shape == (15, 3)

    # Test one iteration
    model.iter_once()
    assert hasattr(model, "obj")
    assert len(model.obj) > 0

    # Test fit method
    result = model.fit(maxit=2)
    # Note: K might be pruned during fitting, so check that shapes are consistent
    assert result.L.shape[0] == 20  # rows should match
    assert result.F.shape[0] == 15  # columns should match
    assert result.L.shape[1] == result.F.shape[1]  # factors should have same K
    assert isinstance(result.tau, torch.Tensor)


def test_noise_types():
    """Test different noise types."""
    Y = torch.randn(12, 8)

    # Test constant noise
    model1 = cEBMF(data=Y, noise_type=NoiseType.CONSTANT)
    assert model1.noise.type == NoiseType.CONSTANT

    # Test row-wise noise
    model2 = cEBMF(data=Y, noise_type=NoiseType.ROW_WISE)
    assert model2.noise.type == NoiseType.ROW_WISE

    # Test column-wise noise
    model3 = cEBMF(data=Y, noise_type=NoiseType.COLUMN_WISE)
    assert model3.noise.type == NoiseType.COLUMN_WISE


def test_priors():
    """Test different prior combinations."""
    Y = torch.randn(10, 8)

    # Test normal priors
    model1 = cEBMF(data=Y, prior_L="norm", prior_F="norm")
    assert model1.model.prior_L == "norm"
    assert model1.model.prior_F == "norm"

    # Test mixed priors
    model2 = cEBMF(data=Y, prior_L="exp", prior_F="laplace")
    assert model2.model.prior_L == "exp"
    assert model2.model.prior_F == "laplace"

    # Test laplace priors
    model3 = cEBMF(data=Y, prior_L="laplace", prior_F="laplace")
    assert model3.model.prior_L == "laplace"
    assert model3.model.prior_F == "laplace"
