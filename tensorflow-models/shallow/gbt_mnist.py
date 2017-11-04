import argparse
import functools
import numpy as np
import tensorflow as tf
from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeEstimator
from tensorflow.contrib.boosted_trees.proto import learner_pb2
from tensorflow.contrib.boosted_trees.estimator_batch import custom_loss_head
from tensorflow.contrib.boosted_trees.python.utils import losses
from tensorflow.contrib import metrics as metrics_lib
from tensorflow.python.ops import math_ops
tf.logging.set_verbosity(tf.logging.INFO)


# Prepares eval metrics for multiclass eval
def _multiclass_metrics(predictions, labels, weights):
    metrics = dict()
    logits = predictions["scores"]
    classes = math_ops.argmax(logits, 1)
    metrics["accuracy"] = metrics_lib.streaming_accuracy(classes, labels, weights)
    return metrics


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",type=int,default=1000)
parser.add_argument("--depth", type=int, default=4, help="Maximum depth of weak learners.")
parser.add_argument("--l2", type=float, default=1.0, help="l2 regularization per batch.")
parser.add_argument("--learning_rate",type=float,default=0.1)
parser.add_argument("--examples_per_layer", type=int, default=1000)
parser.add_argument("--num_trees", type=int, default=100)
parser.add_argument("--num_classes", type=int, default=10)
args = parser.parse_args()

learner_config = learner_pb2.LearnerConfig()
learner_config.learning_rate_tuner.fixed.learning_rate = args.learning_rate
learner_config.num_classes = args.num_classes
learner_config.regularization.l1 = 0.0
learner_config.regularization.l2 = args.l2 / args.examples_per_layer
learner_config.constraints.max_tree_depth = args.depth
learner_config.growing_mode = learner_pb2.LearnerConfig.LAYER_BY_LAYER
learner_config.multi_class_strategy = learner_pb2.LearnerConfig.DIAGONAL_HESSIAN

head = custom_loss_head.CustomLossHead(
      loss_fn=functools.partial(losses.per_example_maxent_loss, num_classes=args.num_classes),
      link_fn=tf.identity,
      logit_dimension=args.num_classes,
      metrics_fn=_multiclass_metrics)

estimator = GradientBoostedDecisionTreeEstimator(
    learner_config=learner_config,
    head=head,
    examples_per_layer=args.examples_per_layer,
    num_trees=args.num_trees,
    center_bias=False)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = (X_train / 255.).reshape(-1, 28*28).astype(np.float32)
X_test = (X_test / 255.).reshape(-1, 28*28).astype(np.float32)
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

estimator.fit(input_fn=tf.estimator.inputs.numpy_input_fn(
    x={'_':X_train}, y=y_train, batch_size=args.batch_size, num_epochs=10, shuffle=True))

estimator.evaluate(input_fn=tf.estimator.inputs.numpy_input_fn(
    x={'_':X_test}, y=y_test, batch_size=args.batch_size, shuffle=False))