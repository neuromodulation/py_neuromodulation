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
        ,is_training = True):

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




class TabNet(k.Model):
    def __init__(self,feature_dim, batch_momentum, virtual_batch_size,
                 num_decision_steps, num_features, output_dim,
                 relaxation_factor, batch_normalize_input=True,
                 tensor_image = False, epsilon =0.00001,  **kwargs):

        super(TabNet, self).__init__()
        self.feature_dim = feature_dim
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_decision_steps = num_decision_steps
        self.output_dim = output_dim
        self.num_features = num_features
        self.relaxation_factor = relaxation_factor
        self.batch_normalize_input = batch_normalize_input
        self.tensor_image = tensor_image
        self.epsilon = epsilon
        self.shared_b = self.shared_block


    def complete_init(self):

        self.output_aggregated = tf.zeros((self.batch_size, self.output_dim))
        self.aggregated_mask_values = tf.zeros((self.batch_size, self.num_features))
        self.mask_values = tf.zeros((self.batch_size, self.num_features))
        self.complemantary_aggregated_mask_values = tf.ones(
            [self.batch_size, self.num_features])
        self.total_entropy = 0


    def glu(self, act, n_units):
      """Generalized linear unit nonlinear activation."""
      return act[:, :n_units] * k.activations.sigmoid(act[:, n_units:])


    def shared_block(self,input):
        transform1 = k.layers.Dense(self.feature_dim*2, name = "Transform1",use_bias=False)(input)
        transform1 = k.layers.BatchNormalization(momentum=self.batch_momentum, virtual_batch_size=self.virtual_batch_size)(transform1)
        out1 = self.glu(transform1, self.feature_dim)

        transform2 = k.layers.Dense(self.feature_dim*2, name = "Transform2",use_bias=False)(out1)
        transform2 = k.layers.BatchNormalization(momentum=self.batch_momentum, virtual_batch_size=self.virtual_batch_size)(transform2)
        out2 = (self.glu(transform2, self.feature_dim) + out1)* tf.math.sqrt(0.5) # This could be a place where important information is lost

        return out2


    def feature_transformer(self, input, i):
        transform3 = k.layers.Dense(self.feature_dim*2,name="transform3_"+str(i) )(input)
        transform3 = k.layers.BatchNormalization(momentum=self.batch_momentum, virtual_batch_size=self.virtual_batch_size)(transform3)
        out3 = (self.glu(transform3,self.feature_dim)+input) * tf.math.sqrt(0.5)

        transform4 = k.layers.Dense(self.feature_dim*2,name="transform3_"+str(i) )(out3)
        transform4 = k.layers.BatchNormalization(momentum=self.batch_momentum, virtual_batch_size=self.virtual_batch_size)(transform4)
        out4 = (self.glu(transform4,self.feature_dim)+out3) * tf.math.sqrt(0.5)

        return out4

    def split(self, transformed_features):

        decision_out = k.layers.ReLU()(transformed_features[:, :self.output_dim])

        self.output_aggregated += decision_out

        scale_agg = tf.reduce_sum(
            decision_out, axis=1, keepdims=True) / (self.num_decision_steps-1)

        self.aggregated_mask_values += self.mask_values * scale_agg

    def mask_update(self,coef_features, ni):

        mask_values = k.layers.Dense(self.num_features,
                                     name = 'transform_coeff'+str(ni),
                                     use_bias=False)(coef_features)
        mask_values = k.layers.BatchNormalization(momentum=self.batch_momentum,
                                                  virtual_batch_size=self.virtual_batch_size)(mask_values)

        mask_values *= self.complemantary_aggregated_mask_values
        mask_values = tfa.layers.Sparsemax()(mask_values)

        self.mask_values = mask_values

        self.complemantary_aggregated_mask_values *= (self.relaxation_factor - mask_values)

        self.total_entropy += tf.reduce_mean(
            tf.reduce_sum(
                -mask_values * tf.math.log(mask_values+self.epsilon),
                axis=1)) / (self.num_decision_steps-1)

        masked_features =  k.layers.Multiply()([mask_values,self.features])

        return masked_features

    def call(self, input):

        if self.batch_normalize_input:
            self.features = k.layers.BatchNormalization()(input)
        else:
            self.features = input

        self.batch_size = tf.shape(self.features)[0]
        self.complete_init()

        masked_features = self.features

        for ni in range(self.num_decision_steps):

            shared = self.shared_b(masked_features)
            transformed_f = self.feature_transformer(shared, ni)

            if ni > 0 :
                self.split(transformed_f)

            features_for_coef = (transformed_f[:, self.output_dim:])

            if ni < self.num_decision_steps-1:
                masked_features = self.mask_update(features_for_coef,ni)

                if self.tensor_image:
                 tf.summary.image("Mask for step"+str(ni),
                                  tf.expand_dims(tf.expand_dims(self.mask_values,0),3), max_outputs=1)


        if self.tensor_image:
            tf.summary.image("Aggregated mask",
                             tf.expand_dims(tf.expand_dims(self.aggregated_mask_values,0),3),max_outputs=1)

        return k.layers.Dense(self.output_dim,activation='linear',use_bias=False)(self.output_aggregated)


