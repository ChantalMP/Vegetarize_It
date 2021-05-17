from flask import render_template

from app import app
from app.forms import SubmitForm, ResetForm
from main import Recipe_Vegeterizer

vegeterizer = Recipe_Vegeterizer()

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    submit_form = SubmitForm()
    reset_form = ResetForm()
    if submit_form.validate_on_submit():
        old_recipe = submit_form.recipe.data
        new_recipe, replacements = vegeterizer.vegeterize(old_recipe)
        replacements = [{'ingr': key, 'r': val} for key, val in replacements.items()]
        return render_template('index.html', new_recipe=new_recipe, replacements=replacements, reset_form=reset_form, old_recipe=old_recipe)
    return render_template('index.html', submit_form=submit_form)