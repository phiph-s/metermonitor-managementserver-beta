import numpy as np
import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class CostSensitiveLoss(tf.keras.losses.Loss):
    def __init__(self, r_idx, n_classes, penalty_weight=0.2, name='CostSensitiveLoss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.r_idx = int(r_idx)
        self.n_classes = int(n_classes)
        self.penalty_weight = float(penalty_weight)

        # Build cost matrix
        C_np = np.ones((self.n_classes, self.n_classes), dtype=np.float32) * 5.0
        np.fill_diagonal(C_np, 0.0)

        digit_indices = [i for i in range(self.n_classes) if i != self.r_idx]
        for d in digit_indices:
            C_np[d, self.r_idx] = 1.5
        for d in digit_indices:
            C_np[self.r_idx, d] = 10.0

        self.C_tf = tf.constant(C_np)

    def call(self, y_true, y_pred):
        cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        C_row = tf.matmul(y_true, self.C_tf)
        penalty = tf.reduce_sum(y_pred * C_row, axis=1)
        return cce + self.penalty_weight * penalty

    def get_config(self):
        base = super().get_config()
        base.update({
            'r_idx': int(self.r_idx),
            'n_classes': int(self.n_classes),
            'penalty_weight': float(self.penalty_weight)
        })
        return base

    @classmethod
    def from_config(cls, config):
        return cls(**config)
