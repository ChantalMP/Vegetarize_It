from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired
from wtforms.widgets import TextArea


class SubmitForm(FlaskForm):
    recipe = StringField('Recipe', widget=TextArea(), validators=[DataRequired()])
    vegetarize = SubmitField('Vegetarize')

class ResetForm(FlaskForm):
    reset = SubmitField('Input new recipe')