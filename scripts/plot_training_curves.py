# coding=utf-8
"""
Plot rotation_error curves from TensorBoard event files for baseline vs CVT.
Usage:
  python scripts/plot_training_curves.py \
    --baseline_logdir path/to/baseline \
    --cvt_logdir path/to/cvt \
    --out_png eval_summary/rotation_error_comparison.png
"""
from absl import app, flags
import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

flags.DEFINE_string('baseline_logdir', '', 'TB logdir for baseline (siamese)')
flags.DEFINE_string('cvt_logdir', '', 'TB logdir for CVT encoder')
flags.DEFINE_string('out_png', 'eval_summary/rotation_error_comparison.png', 'Output plot path')
flags.DEFINE_string('tag', 'rotation_error', 'Scalar tag to plot (in degrees)')
FLAGS = flags.FLAGS


def load_scalar(logdir, tag):
  if not logdir or not os.path.exists(logdir):
    return []
  ea = event_accumulator.EventAccumulator(logdir)
  ea.Reload()
  if tag not in ea.Tags().get('scalars', []):
    return []
  evts = ea.Scalars(tag)
  return [(e.step, e.value) for e in evts]


def main(_):
  base = load_scalar(FLAGS.baseline_logdir, FLAGS.tag)
  cvt = load_scalar(FLAGS.cvt_logdir, FLAGS.tag)
  plt.figure(figsize=(8,4))
  if base:
    steps, vals = zip(*base)
    plt.plot(steps, vals, label='Baseline (Siamese)')
  if cvt:
    steps, vals = zip(*cvt)
    plt.plot(steps, vals, label='CVT Encoder')
  plt.xlabel('Step')
  plt.ylabel(FLAGS.tag + ' (deg)')
  plt.title('Rotation error vs step')
  plt.legend()
  os.makedirs(os.path.dirname(FLAGS.out_png), exist_ok=True)
  plt.tight_layout()
  plt.savefig(FLAGS.out_png)
  print('Saved plot to', FLAGS.out_png)

if __name__ == '__main__':
  app.run(main)
