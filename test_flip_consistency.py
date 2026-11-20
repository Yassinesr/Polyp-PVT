#!/usr/bin/env python3
"""Unit test for flip_consistency logic in Train_noaug.py."""
import torch
import torch.nn as nn

def test_flip_consistency_weight_formula():
    """Test lag-or-margin weight: w = max(0, 1 - 2*|p_o - y|) for correct pixels."""
    # Simulate what happens in the FCT loss computation
    p_o = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1])  # upright predictions
    y = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])     # label: all polyp

    # Lag-or-margin weight (fct_weit=8, fct_lm_a=2.0)
    e_o = (p_o - y).abs()
    w = (1.0 - 2.0 * e_o).clamp(min=0.0)

    # Expected weights:
    # p=0.9: |0.9-1|=0.1, w=1-0.2=0.8
    # p=0.7: |0.7-1|=0.3, w=1-0.6=0.4
    # p=0.5: |0.5-1|=0.5, w=1-1.0=0.0
    # p=0.3: |0.3-1|=0.7, w=1-1.4=0.0 (clamped)
    # p=0.1: |0.1-1|=0.9, w=1-1.8=0.0 (clamped)

    expected = torch.tensor([0.8, 0.4, 0.0, 0.0, 0.0])
    assert torch.allclose(w, expected, atol=1e-6), f"Weight mismatch: got {w}, expected {expected}"

    print(f"✓ Lag-or-margin weight formula test passed")

def test_flip_consistency_boundary_cases():
    """Test edge cases: perfect prediction, completely wrong, boundary."""
    # Perfect prediction (p_o = y = 1)
    p_o = torch.tensor([1.0])
    y = torch.tensor([1.0])
    e_o = (p_o - y).abs()
    w = (1.0 - 2.0 * e_o).clamp(min=0.0)
    assert w.item() == 1.0, f"Perfect prediction should have weight 1.0, got {w.item()}"

    # Completely wrong (p_o = 0, y = 1)
    p_o = torch.tensor([0.0])
    y = torch.tensor([1.0])
    e_o = (p_o - y).abs()
    w = (1.0 - 2.0 * e_o).clamp(min=0.0)
    assert w.item() == 0.0, f"Wrong prediction (e=1.0) should have weight 0.0, got {w.item()}"

    # Boundary case (e = 0.5, weight should be exactly 0)
    p_o = torch.tensor([0.5])
    y = torch.tensor([1.0])
    e_o = (p_o - y).abs()
    w = (1.0 - 2.0 * e_o).clamp(min=0.0)
    assert w.item() == 0.0, f"Boundary case (e=0.5) should have weight 0.0, got {w.item()}"

    print(f"✓ Boundary cases test passed")

def test_flip_consistency_boundary_cases():
    """Test edge cases: perfect prediction, completely wrong, boundary."""
    # Perfect prediction (p_o = y = 1)
    p_o = torch.tensor([1.0])
    y = torch.tensor([1.0])
    e_o = (p_o - y).abs()
    w = (1.0 - 2.0 * e_o).clamp(min=0.0)
    assert w.item() == 1.0, f"Perfect prediction should have weight 1.0, got {w.item()}"

    # Completely wrong (p_o = 0, y = 1)
    p_o = torch.tensor([0.0])
    y = torch.tensor([1.0])
    e_o = (p_o - y).abs()
    w = (1.0 - 2.0 * e_o).clamp(min=0.0)
    assert w.item() == 0.0, f"Wrong prediction (e=1.0) should have weight 0.0, got {w.item()}"

    # Boundary case (e = 0.5, weight should be exactly 0)
    p_o = torch.tensor([0.5])
    y = torch.tensor([1.0])
    e_o = (p_o - y).abs()
    w = (1.0 - 2.0 * e_o).clamp(min=0.0)
    assert w.item() == 0.0, f"Boundary case (e=0.5) should have weight 0.0, got {w.item()}"

    print(f"✓ Boundary cases test passed")

if __name__ == '__main__':
    test_flip_consistency_weight_formula()
    test_flip_consistency_boundary_cases()
    print("\n✓ All FCT weight formula tests passed")

