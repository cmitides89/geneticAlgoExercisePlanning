# import new_ga
import genetic_class as gc
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, RadioField
from wtforms.validators import Required, InputRequired
# NOTE: RENDER TEMPLATE EXPECTS TEMPLATES TO BE SOTRED IN Templates
# folder in same level as app.py
app = Flask('__name__')
# TODO: look at chap 7 on how to embed key in env var 
app.config['SECRET_KEY'] = 'hard to guess string'

class RequirementsForm(FlaskForm):
    no_days = RadioField('Number of days', choices=[('2', '2 days a week'),
                                                    ('3', '3 days a week'),
                                                    ('4', '4 days a week'),
                                                    ('5', '5 days a week')], validators=[InputRequired()])
    lvl_exp = RadioField('Level of experience', choices=[('beginner', 'Beginner'), (
        'intermediate', 'Intermediate'), ('advanced', 'Advanced')], validators=[InputRequired()])
    goal_type = RadioField('Goal', choices=[(
        'Hypertrophy', 'Hypertrophy'), ('Strength', 'Strength')], validators=[InputRequired()])
    no_ex = RadioField('Number of Exercises', choices=[('4','4'),('5','5'),('6','6'),('7','7')], validators=[InputRequired()])

# routes set up
@app.route('/', methods = ['POST','GET'])
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
        result_list = gen_c.create_day_pheno(gene_type='lowerbody')
        print(result_list)
        return render_template('exercise_result.html', results = result_list)
    return render_template('exerciseform.html', form=ga_input)

# @app.route('/results', methods=['POST'])
# def results():
#     form = request.form
#     if request.method == 'POST':
#         # model = get_model()

            
