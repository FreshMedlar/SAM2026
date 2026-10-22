import torch
import torch.nn.functional as F

def compute_cosine_similarity(tensor1, tensor2, eps=1e-8):
    """Compute cosine similarity between two tensors flattened."""
    t1 = tensor1.detach().flatten()
    t2 = tensor2.detach().flatten()
    norm1 = torch.norm(t1)
    norm2 = torch.norm(t2)
    if norm1 < eps or norm2 < eps:
        return 0.0
    return (torch.dot(t1, t2) / (norm1 * norm2)).item()

def compute_norm_ratio(tensor_test, tensor_ref, eps=1e-8):
    """Compute relative norm ||tensor_test|| / ||tensor_ref||."""
    norm_test = torch.norm(tensor_test.detach())
    norm_ref = torch.norm(tensor_ref.detach())
    return (norm_test / (norm_ref + eps)).item()

def compute_weight_alignment(model_forward, model_feedback):
    """
    Compute weight alignment CosineSim(W_forward, W_feedback^T) across layers.
    Returns per-layer similarities and the mean similarity.
    """
    fwd_layers = model_forward.get_linear_layers()
    fb_layers = model_feedback.get_linear_layers()
    layer_alignments = []

    for f_layer, b_layer in zip(fwd_layers, fb_layers):
        w_f = f_layer.w.weight.data
        w_b = b_layer.w.weight.data  # same shape [out_dim, in_dim]
        # Compare W_f and W_b (they have identical shape)
        align = compute_cosine_similarity(w_f, w_b)
        layer_alignments.append(align)

    mean_align = sum(layer_alignments) / len(layer_alignments)
    return layer_alignments, mean_align

def compute_gradient_similarity(model_synaptic, X, y_one_hot, loss_fn):
    """
    Calculate the cosine similarity and norm ratio between the synaptic gradients
    and the true analytical autograd gradients on the exact same mini-batch.
    """
    # 1. Compute true autograd gradients for model_synaptic
    model_synaptic.zero_grad()
    x_cloned = X.clone().detach().requires_grad_(False)
    out_bp = model_synaptic(x_cloned, x_masks=None, nonlin=F.relu)
    loss_true_bp = loss_fn(out_bp, y_one_hot)
    loss_true_bp.backward()

    true_grads = []
    for layer in model_synaptic.get_linear_layers():
        true_grads.append(layer.w.weight.grad.clone().detach())

    # 2. Get synaptic gradients stored in layer.w.weight.grad
    syn_grads = []
    for layer in model_synaptic.get_linear_layers():
        syn_grads.append(layer.w.weight.grad.detach())

    # Flatten across all layers
    flat_true = torch.cat([g.flatten() for g in true_grads])
    flat_syn = torch.cat([g.flatten() for g in syn_grads])

    cos_sim = compute_cosine_similarity(flat_syn, flat_true)
    norm_ratio = compute_norm_ratio(flat_syn, flat_true)

    # Clean up gradients
    model_synaptic.zero_grad()

    return cos_sim, norm_ratio
