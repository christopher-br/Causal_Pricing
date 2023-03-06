# Pipeline by Christopher Bockel-Rickermann. Copyright (c) 2022

# Import necessary modules
import numpy as np
import pandas as pd

from modules.data_prepper import get_observation_outcome

import plotly.graph_objects as go

def print_summary_plot(true_dr, mlp_dr, drnet_dr, linreg_dr, logreg_dr, rf_dr, treatment_strengths, dataset, id, bias, name):    
    # Save treatment strengths
    samples_power_of_two = 6
    num_integration_samples = 2 ** samples_power_of_two + 2
    bins = np.linspace(np.finfo(float).eps, 0.999, num_integration_samples)
    
    # Bars for density
    bars = np.histogram(dataset['d'], bins)[0]
    bars = (bars)/np.max(bars)
    
    data = {'xaxis': treatment_strengths,
            'd': bars,
            'True response': true_dr,
            'MLP': mlp_dr,
            'DRNets': drnet_dr,
            'Linear reg.': linreg_dr,
            'Logistic reg.': logreg_dr,
            'Random forest': rf_dr}
    df = pd.DataFrame(data)
    
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=df['xaxis'], y=df['True response'], line_shape='linear', name='True response'))
    figure.add_trace(go.Scatter(x=df['xaxis'], y=df['MLP'], line_shape='linear', name='MLP'))
    figure.add_trace(go.Scatter(x=df['xaxis'], y=df['DRNets'], line_shape='linear', name='DRNets'))
    figure.add_trace(go.Scatter(x=df['xaxis'], y=df['Linear reg.'], line_shape='linear', name='Linear reg.'))
    figure.add_trace(go.Scatter(x=df['xaxis'], y=df['Logistic reg.'], line_shape='linear', name='Logistic reg.'))
    figure.add_trace(go.Scatter(x=df['xaxis'], y=df['Random forest'], line_shape='linear', name='Random forest'))
    figure.add_trace(go.Bar(x=df['xaxis'], y=df['d'], name="Relative density", opacity=0.5, marker_color='lightblue'))
    
    figure.update_layout(
        barmode='group', 
        bargap=0.0, 
        bargroupgap=0.0, 
        paper_bgcolor='white', 
        plot_bgcolor='white',
        shapes=[go.layout.Shape(type='rect', 
                                xref='paper', yref='paper', 
                                x0=0.0, y0=0.0, 
                                x1=1.0, y1=1.0,
                                line={'width': 1, 'color': 'black'})],
        title=("Summary | bias: " + str(bias)),
        xaxis_title="Dosage",
        yaxis_title="Probability",
        legend_title="Models")
    
    figure.update_yaxes(range=[0,1])
    
    # Return figure
    return figure

def get_dr_SKL(observation, model, type="Regressor"):
    # Save treatment strengths
    samples_power_of_two = 6
    num_integration_samples = 2 ** samples_power_of_two + 1
    treatment_strengths = np.linspace(np.finfo(float).eps, 0.999, num_integration_samples)
    
    # Save test data
    test_data = dict()
    test_data['x'] = np.repeat(np.expand_dims(observation, axis=0), num_integration_samples, axis=0) # Attach dosages
    test_data['d'] = treatment_strengths
    test_data['xd'] = np.column_stack((test_data['x'], test_data['d']))
    
    # Predict dr curve
    if type=="Regressor":
        dr_curve = model.predict(test_data['xd'])
    elif type=="Classifier":
        dr_curve = model.predict_proba(test_data['xd'])[:,1]
        
    drr_curve = (1 - dr_curve) * treatment_strengths
    
    return dr_curve, drr_curve

def get_true_dr(observation, dataset_test):
    # Save treatment strengths
    samples_power_of_two = 6
    num_integration_samples = 2 ** samples_power_of_two + 1
    treatment_strengths = np.linspace(np.finfo(float).eps, 0.999, num_integration_samples)
    
    treatment_idx = 0 # Only one treatment
    
    dr_curve = np.array([get_observation_outcome(observation, dataset_test['metadata']['v'], treatment_idx, d) for d in treatment_strengths])
    
    drr_curve = (1 - dr_curve) * treatment_strengths
    
    return dr_curve, drr_curve


def get_treatment_strengths():
    # Save treatment strengths
    samples_power_of_two = 6
    num_integration_samples = 2 ** samples_power_of_two + 1
    treatment_strengths = np.linspace(np.finfo(float).eps, 0.999, num_integration_samples)
    
    return treatment_strengths