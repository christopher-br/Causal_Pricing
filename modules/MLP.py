# SCIGAN implementation based on code by Ioana Bica (2020)
# Remaining pipeline by Christopher Bockel-Rickermann. Copyright (c) 2022
# Changes: Removed generator and discriminator
# Set number dosage samples to 1 to stabilize NN architecture

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import shutil
import os

from utils.model_utils import sample_dosages, sample_X
from utils.eval_utils import compute_val_SCIMLP


class SCIMLP_Model:
    def __init__(self, params):
        self.num_features = params['num_features']
        self.num_treatments = params['num_treatments']
        self.export_model_dir = params['export_model_dir']

        self.h_dim = params['h_dim']
        self.batch_size = params['batch_size']
        self.alpha = params['alpha']
        self.num_dosage_samples = 1 # instead of params['num_dosage_samples']
        
        self.num_iter_inference = params['num_iter_inference']

        self.size_z = self.num_treatments
        self.num_outcomes = self.num_treatments

        tf.reset_default_graph()
        tf.random.set_random_seed(10)

        # Feature (X)
        self.X = tf.placeholder(tf.float32, shape=[None, self.num_features], name='input_features')
        # Treatment (T) - one-hot encoding for the treatment
        self.T = tf.placeholder(tf.float32, shape=[None, self.num_treatments], name='input_treatment')
        # Dosage (D)
        self.D = tf.placeholder(tf.float32, shape=[None, 1], name='input_dosage')
        # Dosage samples (D)
        self.Treatment_Dosage_Samples = tf.placeholder(tf.float32,
                                                       shape=[None, self.num_treatments, 1],
                                                       name='input_treatment_dosage_samples')
        # Treatment dosage mask to indicate the factual outcome
        self.Treatment_Dosage_Mask = tf.placeholder(tf.float32,
                                                    shape=[None, self.num_treatments, 1],
                                                    name='input_treatment_dosage_mask')
        # Outcome (Y)
        self.Y = tf.placeholder(tf.float32, shape=[None, 1], name='input_y')
        # Removed Random Noise (G)
        
    #Removed Generator
    #Removed Treatment Discriminator
    #Removed Dosage Discriminator

    def inference(self, x, treatment_dosage_samples):
        with tf.variable_scope('inference', reuse=tf.AUTO_REUSE):
            inputs = x
            I_shared = tf.layers.dense(inputs, self.h_dim, activation=tf.nn.elu, name='shared')

            I_treatment_dosage_outcomes = dict()

            for treatment in range(self.num_treatments):
                dosage_counterfactuals = dict()
                treatment_dosages = treatment_dosage_samples[:, treatment]

                dosage_sample = tf.expand_dims(treatment_dosages[:, 0], axis=-1)
                input_counterfactual_dosage = tf.concat(axis=1, values=[I_shared, dosage_sample])

                treatment_layer_1 = tf.layers.dense(input_counterfactual_dosage, self.h_dim, activation=tf.nn.elu,
                                                    name='treatment_layer_1_%s' % str(treatment),
                                                    reuse=tf.AUTO_REUSE)

                treatment_layer_2 = tf.layers.dense(treatment_layer_1, self.h_dim, activation=tf.nn.elu,
                                                    name='treatment_layer_2_%s' % str(treatment),
                                                    reuse=tf.AUTO_REUSE)
                
                # The output layer
                treatment_dosage_output = tf.layers.dense(treatment_layer_2, 1, activation=None,
                                                            name='treatment_output_%s' % str(treatment),
                                                            reuse=tf.AUTO_REUSE)

                dosage_counterfactuals[0] = treatment_dosage_output

                I_treatment_dosage_outcomes[treatment] = tf.concat(list(dosage_counterfactuals.values()), axis=-1)

            I_logits = tf.concat(list(I_treatment_dosage_outcomes.values()), axis=1)
            I_logits = tf.reshape(I_logits, shape=(-1, self.num_treatments, 1))

        return I_logits, I_treatment_dosage_outcomes

    def train(self, Train_X, Train_T, Train_D, Train_Y, verbose=False):
        # Removed counterfactual generator
        # Removed dosage discriminator
        # Removed treatment discriminator

        # 4. Inference network
        I_logits, I_treatment_dosage_outcomes = self.inference(self.X, self.Treatment_Dosage_Samples)

        # Removed G_outcomes
        I_outcomes = tf.identity(I_logits, name="inference_outcomes")
        
        # Removed dosage discriminator loss
        # Removed treatment discriminator loss
        # Removed overall discriminator loss
        # Removed generator loss

        # 4. Inference loss
        I_logit_factual = tf.expand_dims(tf.reduce_sum(self.Treatment_Dosage_Mask * I_logits, axis=[1, 2]), axis=-1)
        # rm I_loss1 = tf.reduce_mean((G_logits - I_logits) ** 2)
        # rm I_loss2 = tf.reduce_mean((self.Y - I_logit_factual) ** 2)
        # rem I_loss = tf.sqrt(I_loss1) + tf.sqrt(I_loss2)
        # Set I_loss = I_loss2:
        I_loss = tf.reduce_mean((self.Y - I_logit_factual) ** 2)
        
        # Removed theta_G
        # Removed theta_D_dosage
        # Removed theta_D_treatment
        theta_I = tf.trainable_variables(scope='inference')

        # Removed G_solver
        # Removed D_dosage_solver
        # Removed D_treatment_solver
        I_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(I_loss, var_list=theta_I)

        # Setup tensorflow
        tf_device = 'gpu'
        if tf_device == "cpu":
            tf_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 0})
        else:
            tf_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 1})
            tf_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        # Iterations
        # Removed training generator and discriminator        

        # Train Inference Network
        for it in tqdm(range(self.num_iter_inference), leave=False, desc="Training SCIMLP", ncols=125):
            idx_mb = sample_X(Train_X, self.batch_size)
            X_mb = Train_X[idx_mb, :]
            T_mb = np.reshape(Train_T[idx_mb], [self.batch_size, ])
            D_mb = np.reshape(Train_D[idx_mb], [self.batch_size, ])
            Y_mb = np.reshape(Train_Y[idx_mb], [self.batch_size, 1])
            # Removed Z_G_mb
            
            # Ini random dosage vector with dim [batch_size, num_treatments, num_dosage_samples]
            treatment_dosage_samples = sample_dosages(self.batch_size, self.num_treatments, 1)
            factual_dosage_position = np.random.randint(1, size=[self.batch_size])
            treatment_dosage_samples[range(self.batch_size), T_mb, factual_dosage_position] = D_mb

            treatment_dosage_mask = np.zeros(shape=[self.batch_size, self.num_treatments, 1])
            treatment_dosage_mask[range(self.batch_size), T_mb, factual_dosage_position] = 1
            treatment_one_hot = np.sum(treatment_dosage_mask, axis=-1)

            _, I_loss_curr = self.sess.run([I_solver, I_loss],
                                           feed_dict={self.X: X_mb, self.T: treatment_one_hot,
                                                      self.D: D_mb[:, np.newaxis],
                                                      self.Treatment_Dosage_Samples: treatment_dosage_samples,
                                                      self.Treatment_Dosage_Mask: treatment_dosage_mask, self.Y: Y_mb})
                                                      # Removed self.Z_G: Z_G_mb})

            if it % 1000 == 0 and verbose:
                print('Iter: {}'.format(it))
                print('I_loss: {:.4}'.format((I_loss_curr)))
                print()
                
        self.I_logits = I_logits

        tf.compat.v1.saved_model.simple_save(self.sess, export_dir=self.export_model_dir,
                                             inputs={'input_features': self.X,
                                                     'input_treatment_dosage_samples': self.Treatment_Dosage_Samples},
                                             outputs={'inference_outcome': I_logits})
    
    def tune(self, Train_X, Train_T, Train_D, Train_Y, dataset_val, batch_sizes, h_dims):

        best_val_error = np.inf

        # Set parameters:
        params_help = dict()
        params_help['num_treatments'] = self.num_treatments
        params_help['num_features'] = self.num_features
        params_help['num_dosage_samples'] = 1
        params_help['alpha'] = self.alpha
        params_help['num_iter_generator'] = 5000
        params_help['num_iter_inference'] = 10000
        params_help['export_model_dir'] = 'outputs/saved_models/SCIMLP_train'
        
        # For progress tracker
        tuning_loops = len(batch_sizes) * len(h_dims)
        
        # Get progress bar
        pbar = tqdm(desc="Tuning SCIMLP", leave=False, ncols=125, total=tuning_loops)
        
        # Iterate over training parameters
        for batch_size in batch_sizes:
            params_help['batch_size'] = batch_size  
                
            for h_dim in h_dims:
                params_help['h_dim'] = h_dim
                params_help['h_inv_eqv_dim'] = h_dim

                mod = SCIMLP_Model(params_help)
                mod.train(Train_X=Train_X, 
                            Train_T=Train_T, 
                            Train_D=Train_D,
                            Train_Y=Train_Y, 
                            verbose=False)

                # Calculate validation MISE (observed):
                err = compute_val_SCIMLP(dataset_val=dataset_val, 
                                        num_treatments=self.num_treatments,
                                        num_dosage_samples=self.num_dosage_samples,
                                        model_folder=params_help['export_model_dir'])
                
                # Update pbar
                pbar.update(1)
                
                # Check if error is better than current best
                if err < best_val_error:
                    # Save most recent best
                    best_val_error = err
                    
                    # Save current params
                    self.batch_size = batch_size
                    self.h_dim = h_dim
                    self.h_inv_eqv_dim = h_dim

                    # Save model:
                    # Delete current best
                    if os.path.exists(self.export_model_dir):
                        shutil.rmtree(self.export_model_dir)
                    
                    # Load last session
                    with tf.Session(graph=tf.Graph()) as sess:
                        tf.saved_model.loader.load(sess, ["serve"], params_help['export_model_dir'])
                        
                        # Save
                        tf.compat.v1.saved_model.simple_save(sess, export_dir=self.export_model_dir,
                                                                inputs={'input_features': mod.X,
                                                                        'input_treatment_dosage_samples': mod.Treatment_Dosage_Samples},
                                                                outputs={'inference_outcome': mod.I_logits})
                        
                # Delete current model
                if os.path.exists(params_help['export_model_dir']):
                    shutil.rmtree(params_help['export_model_dir'])
                            
        pbar.close()
