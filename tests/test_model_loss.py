"""Unit tests for ResNet-18 CIFAR encoder, projection head, linear classifier, and NT-Xent loss."""
import copy
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.simclr.loss import NTXentLoss, nt_xent_loss_explicit_loop
from src.simclr.model import (
    LinearClassifier,
    ProjectionHead,
    ResNet18CIFAR,
    SimCLRModel,
)


def test_encoder_cifar_stem_and_output_shape():
    """Verify CIFAR stem and output representation h shape (B, 512)."""
    model = ResNet18CIFAR()
    # Ensure no max-pooling layer exists in encoder
    for name, module in model.named_modules():
        assert not isinstance(module, nn.MaxPool2d), f"Found forbidden MaxPool2d in {name}"

    # Ensure conv1 is 3x3 stride 1
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)

    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    h = model(x)
    assert h.shape == (batch_size, 512)
    assert h.dtype == torch.float32


def test_projection_head_shape():
    """Verify projection head output shape (B, 128)."""
    head = ProjectionHead(in_dim=512, hidden_dim=512, out_dim=128)
    h = torch.randn(8, 512)
    z = head(h)
    assert z.shape == (8, 128)
    assert z.dtype == torch.float32


def test_full_simclr_model_forward():
    """Verify full model forward returns both representation and projection."""
    simclr = SimCLRModel()
    x = torch.randn(4, 3, 32, 32)
    h, z = simclr(x)
    assert h.shape == (4, 512)
    assert z.shape == (4, 128)

    # get_representation only computes h
    h_only = simclr.get_representation(x)
    assert torch.allclose(h, h_only, atol=1e-5)


def test_nt_xent_loss_vs_reference_loop():
    """Verify that vectorized NT-Xent loss matches explicit reference loop to tolerance 1e-5."""
    torch.manual_seed(42)
    N = 8
    D = 128
    temperature = 0.5

    z_i = torch.randn(N, D)
    z_j = torch.randn(N, D)

    criterion = NTXentLoss(temperature=temperature)
    vectorized_loss = criterion(z_i, z_j)
    reference_loss = nt_xent_loss_explicit_loop(z_i, z_j, temperature=temperature)

    assert torch.isclose(vectorized_loss, reference_loss, atol=1e-5), (
        f"Vectorized {vectorized_loss.item():.6f} != Reference {reference_loss.item():.6f}"
    )


def test_nt_xent_loss_properties():
    """Verify that identical views have lower loss than random views, and no NaNs across temperatures."""
    torch.manual_seed(42)
    N = 16
    D = 128

    # Random views
    z_i = torch.randn(N, D)
    z_j_random = torch.randn(N, D)
    # Identical views
    z_j_identical = z_i.clone()

    criterion = NTXentLoss(temperature=0.5)
    loss_random = criterion(z_i, z_j_random)
    loss_identical = criterion(z_i, z_j_identical)

    # Perfect agreement must have significantly lower loss than random
    assert loss_identical.item() < loss_random.item()

    # Temperature stability (tau in 0.1 to 1.0)
    for tau in [0.1, 0.2, 0.5, 0.8, 1.0]:
        crit_tau = NTXentLoss(temperature=tau)
        loss_val = crit_tau(z_i, z_j_random)
        assert not torch.isnan(loss_val)
        assert not torch.isinf(loss_val)
        assert loss_val.item() > 0.0


def test_linear_eval_encoder_frozen_invariance():
    """Verify that an optimizer update on linear classifier leaves encoder weights and BN stats unchanged."""
    encoder = ResNet18CIFAR()
    classifier = LinearClassifier(feature_dim=512, num_classes=10)

    # Freeze encoder per AGENTS.md §4.6
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Snapshot initial state
    initial_encoder_params = [p.clone() for p in encoder.parameters()]
    initial_running_mean = [m.running_mean.clone() for m in encoder.modules() if isinstance(m, nn.BatchNorm2d)]
    initial_running_var = [m.running_var.clone() for m in encoder.modules() if isinstance(m, nn.BatchNorm2d)]

    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1)

    # Simulate 3 training steps of linear classifier
    for _ in range(3):
        x = torch.randn(8, 3, 32, 32)
        y = torch.randint(0, 10, (8,))

        with torch.no_grad():
            h = encoder(x)

        logits = classifier(h)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Check encoder parameters unchanged
    for p_init, p_curr in zip(initial_encoder_params, encoder.parameters()):
        assert torch.equal(p_init, p_curr), "Encoder parameter was modified during linear eval!"

    # Check BN running stats unchanged
    current_bns = [m for m in encoder.modules() if isinstance(m, nn.BatchNorm2d)]
    for m_init_mean, m_init_var, m_curr in zip(initial_running_mean, initial_running_var, current_bns):
        assert torch.equal(m_init_mean, m_curr.running_mean), "Encoder BN mean changed!"
        assert torch.equal(m_init_var, m_curr.running_var), "Encoder BN var changed!"
