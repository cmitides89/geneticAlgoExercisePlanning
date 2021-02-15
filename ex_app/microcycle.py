import functools
import json
from flask import(
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from .app_forms import RequirementsForm as rf
from . import genetic_algo as ga
from ex_app.db import get_db

bp = Blueprint('microcycle', __name__, template_folder='templates', url_prefix='/microcycle')

@bp.route('/form', methods=('GET', 'POST'))
def show_exercise_form():
    ga_input = rf()
    print('poop')
    if ga_input.validate_on_submit():
        # call genetic class 
        ga_cyc = ga.Genetic_Class(ga_input.lvl_exp.data,
            ga_input.goal_type.data,
            ga_input.no_days.data,
            ga_input.no_ex.data)
        
        result_list = ga_cyc.start_evolution()
        result_list = json.loads(result_list.to_json(orient='records'))

        return render_template('microcycle/results.html', results = result_list)

    return render_template('microcycle/microform.html', form=ga_input)
