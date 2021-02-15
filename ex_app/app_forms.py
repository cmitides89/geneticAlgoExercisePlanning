from flask_wtf import FlaskForm
from wtforms import RadioField
from wtforms.validators import InputRequired


class RequirementsForm(FlaskForm):
    print('made reqform')
    no_days = RadioField('Number of days',
                         choices=[('2', '2 days a week'),
                                    ('3', '3 days a week'),
                                    ('4', '4 days a week'),
                                    ('5', '5 days a week')],
                                     validators=[InputRequired()])
    lvl_exp = RadioField('experience',
                         choices=[('beginner', 'Beginner'),
                                     ('intermediate', 'Intermediate'),
                                     ('advanced', 'Advanced')],
                                      validators=[InputRequired()])
    goal_type = RadioField('Goal',
                         choices=[
                             ('Hypertrophy', 'Hypertrophy'),
                              ('strength', 'Strength')],
                               validators=[InputRequired()])
    no_ex = RadioField('Number of Exercises',
                         choices=[('4','4'),
                         ('5','5'),
                         ('6','6'),
                         ('7','7')],
                          validators=[InputRequired()])