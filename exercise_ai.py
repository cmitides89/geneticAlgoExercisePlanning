# import new_ga
# TODO: LOOK INTO REDIS TO HOLD DATA FROM GA
import pprint
pp = pprint.PrettyPrinter(indent=4)
import genetic_class as gc
import plan_filter as pf
from flask import Flask, render_template, request, jsonify
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, RadioField
from wtforms.validators import Required, InputRequired
import json
import pandas
from flask_cors import CORS, cross_origin
# NOTE: RENDER TEMPLATE EXPECTS TEMPLATES TO BE SOTRED IN Templates
# folder in same level as app.py
app = Flask('__name__')
# TODO: look at chap 7 on how to embed key in env var 
app.config['SECRET_KEY'] = 'hard to guess string'
CORS(app)

class RequirementsForm(FlaskForm):
    no_days = RadioField('Number of days', choices=[('2', '2 days a week'),
                                                    ('3', '3 days a week'),
                                                    ('4', '4 days a week'),
                                                    ('5', '5 days a week')], validators=[InputRequired()])
    lvl_exp = RadioField('Level of experience', choices=[('beginner', 'Beginner'), (
        'intermediate', 'Intermediate'), ('advanced', 'Advanced')], validators=[InputRequired()])
    goal_type = RadioField('Goal', choices=[(
        'Hypertrophy', 'Hypertrophy'), ('strength', 'Strength')], validators=[InputRequired()])
    no_ex = RadioField('Number of Exercises', choices=[('4','4'),('5','5'),('6','6'),('7','7')], validators=[InputRequired()])

# routes set up
@app.route('/ga_form', methods = ['POST','GET'])
def show_exercise_form():
    # inputs to send to the ai
    # number of days
    # level of experience
    # plan type (strenght, hypertrophy)
    # sugessted number of exercises
    ga_input = RequirementsForm()
    if ga_input.validate_on_submit():
        gen_c =  gc.Genetic_Class(ga_input.lvl_exp.data,
        ga_input.goal_type.data,
        ga_input.no_days.data,
        ga_input.no_ex.data)

        print(gen_c.usr_lvl, gen_c.goal, gen_c.no_days, gen_c.ex_len)
        print('sending to ga')
        result_list = gen_c.start_evolution()
        print('finished ga')
        result_list = json.loads(result_list.to_json(orient='records'))
        
        print(type(result_list))
        pp.pprint(result_list[0])
        print(len(result_list))
        return render_template('exercise_result.html', results = result_list)
    return render_template('exerciseform.html', form=ga_input)
    # ANGULAR INTERGRATION
    # if request.method == 'POST':
    #     inputs = request.json
    #     # print(type(inputs))
    #     # print(inputs)
    #     print('recieved inputs')
    #     gen_c = gc.Genetic_Class(inputs['LevelOfExp'], inputs['goalType'], inputs['NumberOfDays'], inputs['NumberOfExercises'])
    #     result_list = gen_c.start_evolution()
    #     # TODO: Cut out the unnecessary data before sending to front end. you only want to save the top 
    #     # 5 programs
    #     # single_best_plan = pf.limit_one(result_list)
    #     # print(type(single_best_plan))
    #     # return jsonify(json.loads(result_list.to_json(orient='records')))
    #     # print(type(single_best_plan))
    #     # print('jsonified:::::::::: ',type(jsonify(single_best_plan)))
    #     # print(len(single_best_plan))
    #     # print(pf.flatten_microcyc(result_list))
    #     return (jsonify(pf.flatten_microcyc(result_list)))
    # return 'lol'
# @app.route('/results', methods=['POST'])
# def results():
#     form = request.form
#     if request.method == 'POST':
#         # model = get_model()

            
