import genetic_class as gc
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, RadioField
from wtforms.validators import Required, InputRequired
import json
import pandas as pd
import pprint
pp = pprint.PrettyPrinter(indent=4)
def limit_one(result_list):
    '''
    Given the entire resulting dataframe of GA 
    discard unnecessary columns from user view
    compress into one flat JSON vs nested
    WHAT YOU NEED FROM THE MICOCYCL AS A USER:
    1. GOAL/TYPE
    2. EACH WORKING DAY, DAY# AS KEY
    3. DAY TYPE
    4. LIST OF EX PER DAY
        4.1 FROM EX YOU ONLY NEED: EX_NAME, MAIN-MUSCLE-WORKED, REPS, SETS
            EX_EQUIPMENT, EX_TYPE
    '''
    results_df = pd.DataFrame(result_list)
    # print(results_df.columns)
    results_df.sort_values(by=['micro_rating'], ascending=True)
    best_plan = results_df.iloc[0]
    # print('first row best plan : ', best_plan)
    # best_plan_output = best_plan[['workingdays','usr_lvl','micro_type']]
    workingdays = best_plan[['workingdays']].to_dict()
    # workingdays = pd.DataFrame(workingdays)
    microcycl_info = best_plan[['usr_lvl','micro_type']].to_dict()
    best_plan_output = {**workingdays, **microcycl_info}
    # print('in limit one, best plan output is: ', type(best_plan_output))

    return best_plan_output

def flatten_microcyc(result_list):
    '''
    Input: List of microcycles generated from GA
    Output: highest rated plan in a simplified view
    (all features GA requires are removed)
    transforming the data to a much simpler dictionary 
    for easier front-end javascript traversal
    '''
    # convert list of micros into df to order easier
    result_df = pd.DataFrame(result_list)
    result_df.sort_values(by=['micro_rating'], ascending=True)

    # get the first row of DF its the best plan of this generation
    best_micro = result_df.iloc[0]

    # generate the column names for the best microcycle
    # each workingday needs a day_i
    # flatten_microcyc = dict()
    # for i, day in enumerate(best_micro['workingdays']):
        
    #     flatten_microcyc['day'+str(i+1)] = [day['day_type']]
    #     flatten_microcyc['day'+str(i+1)].append({ex.get('ex_name'), 
    #                                         ex.get('ex_equipment'), 
    #                                         ex.get('main-muscle-worked'), 
    #                                         ex.get('reps'), 
    #                                         ex.get('sets')} for ex in day['exercises'])
    # print( 'flat micro: ', flatten_microcyc)
    

        

    
    # print(type(exercises['workingdays']))
    # return_plan = {key: None for key in bmic_keys}
    # NOTE STARTING OVER
    days_list = best_micro['workingdays']

    all_days = dict()
    # for each day in the day list, only get required information on ex
    # for every ex, then put them in a dict where the key is day_i, and add that to the days dict output
    for i, day in enumerate(days_list):
        print('DAY==============', i)
        # holds all exs from one day
        ex_dict = dict()
        # FORMATTING SINGLE EXERCISE
        for exercise in day['exercises']:
            # a single ex modified
            ex_info = {exercise['ex_name']: {'equipment': exercise['ex_equipment'],
                                             'main muscle': exercise['main-muscle-worked'],
                                             'reps': exercise['reps'],
                                             'sets': exercise['sets']
                                             }
            }
            # ADDING SINGLE EXERCISE TO EXERCISE GROUP
            if not ex_dict:
                ex_dict = ex_info
            else:
                ex_dict.update(ex_info)
        # ADDING EXERCISE GROUP TO THE GROUP OF DAYS 
        ex_dict['day type']= day['day_type']
        all_days['day'+str(i+1)] = ex_dict
        print('inside days loop')
        pp.pprint(all_days)
    print('outside of loop')
    pp.pprint(all_days)

