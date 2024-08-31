from unittest.mock import patch

import pytest

from ice.utils import find_device


@patch("torch.backends.mps.is_available")
@patch("torch.backends.mps.is_built")
@patch("torch.cuda.is_available")
def test_device_is_auto_and_mps_available_and_built(
    mock_cuda_is_available,
    mock_is_built,
    mock_is_available,
):
    mock_cuda_is_available.return_value = False
    mock_is_available.return_value = True
    mock_is_built.return_value = True
    assert find_device("auto") == "mps"


@patch("torch.backends.mps.is_available")
@patch("torch.backends.mps.is_built")
@patch("torch.cuda.is_available")
def test_device_is_auto_and_mps_available_but_not_built(
    mock_cuda_is_available,
    mock_is_built,
    mock_is_available,
):
    mock_cuda_is_available.return_value = False
    mock_is_available.return_value = True
    mock_is_built.return_value = False
    with pytest.raises(OSError):
        find_device("auto")


@patch("torch.backends.mps.is_available")
@patch("torch.cuda.is_available")
def test_device_is_auto_and_cuda_available(
    mock_cuda_is_available,
    mock_mps_is_available,
):
    mock_cuda_is_available.return_value = True
    mock_mps_is_available.return_value = False
    assert find_device("auto") == "cuda"


@patch("torch.backends.mps.is_available")
@patch("torch.cuda.is_available")
def test_device_is_auto_and_cuda_not_available(
    mock_cuda_is_available,
    mock_mps_is_available,
):
    mock_cuda_is_available.return_value = False
    mock_mps_is_available.return_value = False
    assert find_device("auto") == "cpu"


def test_device_is_cpu():
    assert find_device("cpu") == "cpu"


def test_device_is_cuda():
    assert find_device("cuda") == "cuda"


def test_device_is_mps():
    assert find_device("mps") == "mps"


def test_device_is_invalid():
    with pytest.raises(ValueError):
        find_device("invalid")
        find_device(None)
