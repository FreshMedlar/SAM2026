import time
import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from models import SynapticNet, StandardBPNet
from metrics import compute_weight_alignment, compute_cosine_similarity, compute_norm_ratio

def one_hot(labels, num_classes=10):
    vec = torch.zeros(len(labels), num_classes, device=labels.device)
    vec.scatter_(1, labels.unsqueeze(1), 1.0)
    return vec

def update_grad_synaptic(model1, model2, X, Y_onehot, loss_fn):
    """
    Synaptic backward pass implementation from the notebook.
    Returns: loss_fwd, loss_feedback, and computed gradients
    """
    x_masks = []
    output = model1(X, x_masks=x_masks)
    output.retain_grad()
    loss = loss_fn(output, Y_onehot)
    loss.backward()

    O = output.grad.detach()
    O2 = model2(X, x_masks=x_masks)
    loss2 = loss_fn(O2, Y_onehot)

    # Backproject error through feedback network
    _ = model2.backproj(O, x_masks=x_masks, nonlin=None)

    # Compute outer product gradient updates
    syn_grads = []
    fwd_layers = model1.get_linear_layers()
    fb_layers = model2.get_linear_layers()

    for t, s in zip(fwd_layers, fb_layers):
        h = t.ha
        gh = s.hb.grad
        grad = gh.T.mm(h)
        t.w.weight.grad = grad
        s.w.weight.grad = grad
        syn_grads.append(grad.clone().detach())

    return loss.item(), loss2.item(), syn_grads

def evaluate(model, dataloader, device, is_synaptic=False):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            if is_synaptic:
                pred = model(X, x_masks=None, nonlin=torch.nn.functional.relu)
            else:
                pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * X.size(0)
            correct += (pred.argmax(dim=1) == y).sum().item()
            total += X.size(0)

    model.train()
    return total_loss / total, 100.0 * correct / total

def run_benchmark(epochs=5, batch_size=256, lr=0.05, seed=42, output_dir="results"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmarks on: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Prepare MNIST Dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()

    # ==========================================
    # 1. TRAIN SYNAPTIC BACKPROPAGATION
    # ==========================================
    print("\n" + "=" * 50)
    print("TRAINING: Synaptic Masked Backpropagation (No Autograd)")
    print("=" * 50)

    torch.manual_seed(seed)
    syn_model1 = SynapticNet(in_d=784, out_d=10).to(device)
    syn_model2 = SynapticNet(in_d=784, out_d=10, nonlin=lambda x: x).to(device)

    opt1 = optim.SGD(syn_model1.parameters(), lr=lr, weight_decay=1e-3, momentum=0.9)
    opt2 = optim.SGD(syn_model2.parameters(), lr=lr, weight_decay=1e-3, momentum=0.9)

    syn_history = {
        "epoch_time": [],
        "fwd_time_ms": [],
        "bwd_time_ms": [],
        "peak_vram_mb": [],
        "train_loss": [],
        "test_loss": [],
        "test_acc": [],
        "cos_sim_to_bp": [],
        "norm_ratio_to_bp": [],
        "weight_alignment": []
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    for epoch in range(epochs):
        epoch_start = time.time()
        fwd_times, bwd_times = [], []

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            Y_onehot = one_hot(y, 10)

            opt1.zero_grad()
            opt2.zero_grad()

            # Measure step timings
            t0 = time.time()
            loss1, loss2, syn_grads = update_grad_synaptic(syn_model1, syn_model2, X, Y_onehot, loss_fn)
            step_duration = (time.time() - t0) * 1000.0  # ms
            bwd_times.append(step_duration)

            # Gradient similarity to true BP (sample every 50 batches)
            if batch_idx % 50 == 0:
                # Calculate true BP gradient for comparison on this exact state
                syn_model1.zero_grad()
                out_bp = syn_model1(X, x_masks=None, nonlin=torch.nn.functional.relu)
                loss_bp = loss_fn(out_bp, Y_onehot)
                loss_bp.backward()
                true_bp_grads = [l.w.weight.grad.clone().detach() for l in syn_model1.get_linear_layers()]

                flat_syn = torch.cat([g.flatten() for g in syn_grads])
                flat_bp = torch.cat([g.flatten() for g in true_bp_grads])

                cos_sim = compute_cosine_similarity(flat_syn, flat_bp)
                norm_rat = compute_norm_ratio(flat_syn, flat_bp)
                syn_history["cos_sim_to_bp"].append(cos_sim)
                syn_history["norm_ratio_to_bp"].append(norm_rat)

                # Reapply synaptic grads to model1
                for layer, g in zip(syn_model1.get_linear_layers(), syn_grads):
                    layer.w.weight.grad = g

            opt1.step()
            opt2.step()

        epoch_time = time.time() - epoch_start
        test_loss, test_acc = evaluate(syn_model1, test_loader, device, is_synaptic=True)
        _, mean_weight_align = compute_weight_alignment(syn_model1, syn_model2)

        peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0

        syn_history["epoch_time"].append(epoch_time)
        syn_history["fwd_time_ms"].append(np.mean(fwd_times) if fwd_times else 0.0)
        syn_history["bwd_time_ms"].append(np.mean(bwd_times))
        syn_history["peak_vram_mb"].append(peak_vram)
        syn_history["train_loss"].append(loss1)
        syn_history["test_loss"].append(test_loss)
        syn_history["test_acc"].append(test_acc)
        syn_history["weight_alignment"].append(mean_weight_align)

        print(f"Epoch {epoch+1:02d}/{epochs} | Time: {epoch_time:.2f}s | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Weight Align: {mean_weight_align:.4f}")

    # ==========================================
    # 2. TRAIN STANDARD BACKPROPAGATION (SGD)
    # ==========================================
    print("\n" + "=" * 50)
    print("TRAINING: Standard Backpropagation with SGD")
    print("=" * 50)

    torch.manual_seed(seed)
    bp_model = StandardBPNet(in_d=784, out_d=10).to(device)
    bp_opt = optim.SGD(bp_model.parameters(), lr=lr, weight_decay=1e-3, momentum=0.9)

    bp_history = {
        "epoch_time": [],
        "step_time_ms": [],
        "peak_vram_mb": [],
        "train_loss": [],
        "test_loss": [],
        "test_acc": []
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        epoch_start = time.time()
        step_times = []

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            bp_opt.zero_grad()
            t0 = time.time()
            out = bp_model(X)
            loss = loss_fn(out, y)
            loss.backward()
            bp_opt.step()
            step_times.append((time.time() - t0) * 1000.0)

        epoch_time = time.time() - epoch_start
        test_loss, test_acc = evaluate(bp_model, test_loader, device, is_synaptic=False)
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0

        bp_history["epoch_time"].append(epoch_time)
        bp_history["step_time_ms"].append(np.mean(step_times))
        bp_history["peak_vram_mb"].append(peak_vram)
        bp_history["train_loss"].append(loss.item())
        bp_history["test_loss"].append(test_loss)
        bp_history["test_acc"].append(test_acc)

        print(f"Epoch {epoch+1:02d}/{epochs} | Time: {epoch_time:.2f}s | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # ==========================================
    # 3. SAVE RESULTS & GENERATE COMPARISON PLOTS
    # ==========================================
    summary = {
        "synaptic": {
            "final_test_acc": syn_history["test_acc"][-1],
            "final_test_loss": syn_history["test_loss"][-1],
            "avg_epoch_time_s": np.mean(syn_history["epoch_time"]),
            "avg_step_time_ms": np.mean(syn_history["bwd_time_ms"]),
            "peak_vram_mb": max(syn_history["peak_vram_mb"]),
            "final_weight_alignment": syn_history["weight_alignment"][-1],
            "mean_cosine_similarity_to_bp": np.mean(syn_history["cos_sim_to_bp"]),
            "mean_norm_ratio_to_bp": np.mean(syn_history["norm_ratio_to_bp"])
        },
        "standard_bp": {
            "final_test_acc": bp_history["test_acc"][-1],
            "final_test_loss": bp_history["test_loss"][-1],
            "avg_epoch_time_s": np.mean(bp_history["epoch_time"]),
            "avg_step_time_ms": np.mean(bp_history["step_time_ms"]),
            "peak_vram_mb": max(bp_history["peak_vram_mb"])
        }
    }

    with open(os.path.join(output_dir, "benchmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # Plot Comparison Charts
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1. Test Accuracy
    axes[0, 0].plot(range(1, epochs + 1), bp_history["test_acc"], 'o-', label="Standard BP (SGD)", color="#1f77b4", lw=2)
    axes[0, 0].plot(range(1, epochs + 1), syn_history["test_acc"], 's-', label="Synaptic Masked BP", color="#ff7f0e", lw=2)
    axes[0, 0].set_title("Test Accuracy (%) vs Epoch")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)

    # 2. Test Loss
    axes[0, 1].plot(range(1, epochs + 1), bp_history["test_loss"], 'o-', label="Standard BP (SGD)", color="#1f77b4", lw=2)
    axes[0, 1].plot(range(1, epochs + 1), syn_history["test_loss"], 's-', label="Synaptic Masked BP", color="#ff7f0e", lw=2)
    axes[0, 1].set_title("Test Loss vs Epoch")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Cross Entropy Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)

    # 3. Gradient Cosine Similarity
    axes[0, 2].plot(syn_history["cos_sim_to_bp"], color="#2ca02c", lw=1.5)
    axes[0, 2].axhline(y=1.0, color='r', linestyle="--", alpha=0.5, label="Exact Alignment (1.0)")
    axes[0, 2].set_title("Gradient Cosine Similarity (Synaptic vs True BP)")
    axes[0, 2].set_xlabel("Evaluation Step (x50 batches)")
    axes[0, 2].set_ylabel("Cosine Similarity")
    axes[0, 2].legend()
    axes[0, 2].grid(True, linestyle="--", alpha=0.6)

    # 4. Weight Alignment (Feedback Alignment)
    axes[1, 0].plot(range(1, epochs + 1), syn_history["weight_alignment"], '^-', color="#9467bd", lw=2)
    axes[1, 0].set_title("Forward-Backward Weight Alignment Cos(W1, W2)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Weight Cosine Sim")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    # 5. Step Latency Comparison
    categories = ["Standard BP (SGD)", "Synaptic Masked BP"]
    step_times = [summary["standard_bp"]["avg_step_time_ms"], summary["synaptic"]["avg_step_time_ms"]]
    bars = axes[1, 1].bar(categories, step_times, color=["#1f77b4", "#ff7f0e"], width=0.5)
    axes[1, 1].set_title("Average Step Execution Time (ms)")
    axes[1, 1].set_ylabel("Milliseconds (Lower is Faster)")
    for bar in bars:
        yval = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f} ms", ha='center', va='bottom')
    axes[1, 1].grid(True, axis="y", linestyle="--", alpha=0.6)

    # 6. Peak Memory Usage
    vram_usages = [summary["standard_bp"]["peak_vram_mb"], summary["synaptic"]["peak_vram_mb"]]
    bars2 = axes[1, 2].bar(categories, vram_usages, color=["#1f77b4", "#ff7f0e"], width=0.5)
    axes[1, 2].set_title("Peak VRAM Allocated (MB)")
    axes[1, 2].set_ylabel("MB Allocated")
    for bar in bars2:
        yval = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f} MB", ha='center', va='bottom')
    axes[1, 2].grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "benchmark_comparison_plot.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()

    print(f"\nBenchmark completed successfully!")
    print(f"Results saved to: {output_dir}/benchmark_summary.json")
    print(f"Comparison plot saved to: {chart_path}")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="Output directory")
    args = parser.parse_args()

    run_benchmark(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, output_dir=args.output_dir)
