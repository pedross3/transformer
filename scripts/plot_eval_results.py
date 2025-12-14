# coding=utf-8
"""
Plot evaluation results from eval_summary TensorBoard event files.
Usage:
  python scripts/plot_eval_results.py \
    --eval_logdir eval_summary \
    --out_png eval_summary/evaluation_results.png
"""
from absl import app, flags
import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

flags.DEFINE_string('eval_logdir', 'eval_summary', 'TB logdir for evaluation results')
flags.DEFINE_string('out_png', 'eval_summary/evaluation_results.png', 'Output plot path')
FLAGS = flags.FLAGS


def load_scalar(logdir, tag):
    """Load scalar values from TensorBoard event file."""
    if not logdir or not os.path.exists(logdir):
        return []
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        return []
    evts = ea.Scalars(tag)
    return [(e.step, e.value) for e in evts]


def main(_):
    # Load all evaluation metrics
    metrics = {
        'X-axis Error': 'eval/angular_error_x_degrees',
        'Y-axis Error': 'eval/angular_error_y_degrees',
        'Z-axis Error': 'eval/angular_error_z_degrees',
        'Rotation Error': 'eval/rotation_error_degrees'
    }
    
    values = {}
    for name, tag in metrics.items():
        data = load_scalar(FLAGS.eval_logdir, tag)
        if data:
            # Take the last value (or average if multiple evaluations)
            values[name] = np.mean([v for _, v in data])
    
    if not values:
        print(f"No evaluation metrics found in {FLAGS.eval_logdir}")
        return
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(values.keys())
    vals = list(values.values())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    bars = ax.bar(names, vals, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, val in zip(bars, vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}°',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Error (degrees)', fontsize=12)
    ax.set_title('Evaluation Results - Angular Errors', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(vals) * 1.15)  # Add 15% headroom
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(FLAGS.out_png) or '.', exist_ok=True)
    plt.savefig(FLAGS.out_png, dpi=150, bbox_inches='tight')
    print(f'✓ Saved evaluation plot to {FLAGS.out_png}')
    
    # Print summary
    print('\n' + '='*50)
    print('EVALUATION RESULTS SUMMARY')
    print('='*50)
    for name, val in values.items():
        print(f'{name:20s}: {val:7.2f}°')
    print('='*50)


if __name__ == '__main__':
    app.run(main)
