# SCIGAN implementation based on code by Ioana Bica (2020)
# Remaining pipeline by Christopher Bockel-Rickermann. Copyright (c) 2022

import numpy as np
from scipy.integrate import romb
import tensorflow as tf
from tqdm import tqdm

from utils.eval_utils import get_model_predictions, get_generator_predictions


def val_MISE_SCIGAN(dataset_val, num_treatments, num_dosage_samples, model_folder):
    mises = []

    samples_power_of_two = 6
    num_integration_samples = 2 ** samples_power_of_two + 1
    step_size = 0.999 / num_integration_samples
    treatment_strengths = np.linspace(
        np.finfo(float).eps, 0.999, num_integration_samples)

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], model_folder)

        # Get the number of observations to iterate over
        num_test_obs = dataset_val['x'].shape[0]
        # Start iterating
        for obs_id in tqdm(range(num_test_obs), leave=False, desc="Evaluate validation observations", ncols=125):
            # Get the observation
            observation = dataset_val['x'][obs_id]
            for treatment_idx in range(num_treatments):
                val_data = dict()
                val_data['x'] = np.repeat(np.expand_dims(
                    observation, axis=0), num_integration_samples, axis=0)
                val_data['t'] = np.repeat(
                    treatment_idx, num_integration_samples)
                val_data['d'] = treatment_strengths

                # Get predicted dose response curve
                pred_dose_response = get_model_predictions(sess=sess,
                                                           num_treatments=num_treatments,
                                                           num_dosage_samples=num_dosage_samples,
                                                           test_data=val_data)
                pred_dose_response = np.array(pred_dose_response)

                # Get generator predictions
                generator_outcome = get_generator_predictions(sess=sess,
                                                              num_treatments=num_treatments,
                                                              num_dosage_samples=num_dosage_samples,
                                                              test_data=val_data)
                generator_outcome = np.array(generator_outcome)

                # Calculate MISE via romb fct and append
                mise = romb((pred_dose_response - generator_outcome)
                            ** 2, dx=step_size)
                mises.append(mise)

    return np.sqrt(np.mean(mises))
