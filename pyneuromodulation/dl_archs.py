import tensorflow.keras as k
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def glu(act, n_units):
  """Generalized linear unit nonlinear activation."""
  return act[:, :n_units] * k.activations.sigmoid(act[:, n_units:])



def tabnet_regression(num_features = 128
        ,output_dim=1
        , feature_dim=128
        ,num_decision_steps=6
        ,relaxation_factor=1.5
        ,batch_momentum=0.7
        ,epsilon=0.00001
        ,BATCH_SIZE = 4000
        ,virtual_batch_size=2000
        ,is_training = True
):

    input_layer = k.Input(shape=(num_features,))
    features = k.layers.BatchNormalization()(input_layer)
    batch_size = BATCH_SIZE
    output_aggregated = tf.zeros((batch_size, output_dim))
    masked_features = features

    mask_values = tf.zeros((batch_size, num_features))
    aggregated_mask_values = tf.zeros((batch_size, num_features))
    complemantary_aggregated_mask_values = tf.ones((batch_size, num_features))
    total_entropy = 0

    if is_training:
        v_b = virtual_batch_size
    else:
        v_b = 1

    shared_transform_f1 = tf.keras.layers.Dense(
        feature_dim * 2,
        name="Transform_f1",
        use_bias=False)

    shared_transform_f2 = tf.keras.layers.Dense(
        feature_dim * 2,
        name="Transform_f2",
        use_bias=False)

    for ni in range(num_decision_steps):
        # Feature transformer with two shared and two decision step dependent
        # blocks is used below.

        reuse_flag = (ni > 0)

        transform_f1 = shared_transform_f1(masked_features)

        transform_f1 = k.layers.BatchNormalization(
            momentum=batch_momentum,
            virtual_batch_size=v_b
        )(transform_f1)

        transform_f1 = glu(transform_f1, feature_dim)

        transform_f2 = shared_transform_f2(transform_f1)
        transform_f2 = k.layers.BatchNormalization(
            # training=is_training,
            momentum=batch_momentum,
            virtual_batch_size=v_b
        )(transform_f2)
        transform_f2 = (glu(transform_f2, feature_dim) +
                        transform_f1) * np.sqrt(0.5)

        transform_f3 = tf.keras.layers.Dense(
            feature_dim * 2,
            name="Transform_f3" + str(ni),
            use_bias=False)(transform_f2)
        transform_f3 = k.layers.BatchNormalization(
            # training=is_training,
            momentum=batch_momentum,
            virtual_batch_size=v_b
        )(transform_f3)
        transform_f3 = (glu(transform_f3, feature_dim) +
                        transform_f2) * np.sqrt(0.5)

        transform_f4 = tf.keras.layers.Dense(
            feature_dim * 2,
            name="Transform_f4" + str(ni),
            use_bias=False)(transform_f3)
        transform_f4 = k.layers.BatchNormalization(
            momentum=batch_momentum,
            virtual_batch_size=v_b
        )(transform_f4)
        transform_f4 = (glu(transform_f4, feature_dim) +
                        transform_f3) * np.sqrt(0.5)

        if ni > 0:
            decision_out = k.layers.ReLU()(transform_f4[:, :output_dim])

            # Decision aggregation.
            output_aggregated += decision_out

            # Aggregated masks are used for visualization of the
            # feature importance attributes.
            scale_agg = tf.reduce_sum(
                decision_out, axis=1, keepdims=True) / (
                                num_decision_steps - 1)
            aggregated_mask_values += mask_values * scale_agg

        features_for_coef = (transform_f4[:, output_dim:])

        if ni < num_decision_steps - 1:
            # Determines the feature masks via linear and nonlinear
            # transformations, taking into account of aggregated feature use.
            mask_values = k.layers.Dense(
                num_features,
                name="Transform_coef" + str(ni),
                use_bias=False)(features_for_coef)
            mask_values = k.layers.BatchNormalization(
                momentum=batch_momentum,
                virtual_batch_size=v_b
            )(mask_values)
            mask_values *= complemantary_aggregated_mask_values
            mask_values = tfa.layers.Sparsemax()(mask_values)

            complemantary_aggregated_mask_values *= (
                    relaxation_factor - mask_values)

            # Entropy is used to penalize the amount of sparsity in feature
            # selection.
            total_entropy += tf.reduce_mean(
                tf.reduce_sum(
                    -mask_values * tf.math.log(mask_values + epsilon),
                    axis=1)) / (
                                     num_decision_steps - 1)

            # Feature selection.
            masked_features = tf.math.multiply(mask_values, features)

            # tf.summary.image(
            #     "Mask for step" + str(ni),
            #     tf.expand_dims(tf.expand_dims(mask_values, 0), 3),
            #     max_outputs=1)

    # Visualization of the aggregated feature importances
    # tf.summary.image(
    #     "Aggregated mask",
    #     tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3),
    #     max_outputs=1)

    predictions = k.layers.Dense(1, use_bias=False)(output_aggregated)

    model = tf.keras.models.Model(input_layer, predictions)

    # lrs = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # model.compile(k.optimizers.Adam(learning_rate=INIT_LEARNING_RATE),
    #               k.losses.mean_absolute_error,
    #               metrics=[correlation__])

    return model






