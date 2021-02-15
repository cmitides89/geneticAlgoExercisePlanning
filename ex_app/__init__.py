import os
from flask import Flask
from flask_bootstrap import Bootstrap

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'ex_app.sqlite'),
    )
    Bootstrap(app)

    from . import microcycle
    app.register_blueprint(microcycle.bp)

    return app
