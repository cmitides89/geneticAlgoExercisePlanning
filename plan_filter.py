import genetic_class as gc
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, RadioField
from wtforms.validators import Required, InputRequired
import json
import pandas as pd

def limit_one(result_list):
    '''
    Given the entire resulting dataframe of GA 
    return only the Max score microcycle
    '''
    results_df = pd.DataFrame(result_list)
    print(results_df.columns)
    results_df.sort_values(by=['micro_rating'], ascending=True)
    best_plan = results_df.iloc[0]
    print('first row best plan : ', best_plan)
    # best_plan_output = best_plan[['workingdays','usr_lvl','micro_type']]
    workingdays = best_plan[['workingdays']].to_dict()
    microcycl_info = best_plan[['usr_lvl','micro_type']].to_dict()
    # print(workingdays)

    best_plan_output = {**workingdays, **microcycl_info}
    # print('in limit one, best plan output is: ', type(best_plan_output))

    return best_plan_output
