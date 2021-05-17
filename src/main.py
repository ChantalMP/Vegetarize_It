from Foodbert.normalisation.helpers.recipe_normalizer import RecipeNormalizer
from contextualized import ContextualizedSubstitutes

class Recipe_Vegeterizer:

    def __init__(self):
        with open('data/meats_normalized.txt') as f:
            self.meat_ingredients = f.read().splitlines()
        self.cs = ContextualizedSubstitutes(meat_ingredients=self.meat_ingredients)
        self.instruction_normalizer = RecipeNormalizer()

    def vegeterize(self, recipe):
        new_recipe, substitute_dict = self.cs.generate_substitutes_in_recipe(recipe, prefix='<b>', suffix='</b>')
        print(substitute_dict)
        return new_recipe, substitute_dict


if __name__ == '__main__':

    recipe_text = "Whisk together the yogurt, lemon juice, turmeric, garam masala and cumin in a large bowl. Put the chicken in, and coat with the marinade. Cover, and refrigerate (for up to a day). " \
                  "Add the chicken and marinade to the pan, and cook for 5 minutes, then add the chicken stock. " \
                  "Bring the mixture to a boil, then lower the heat and simmer, uncovered, for approximately 30 minutes. Stir in the cream and tomato paste, and simmer until the chicken is cooked through, approximately 10 to 15 minutes. " \
                  "Add the almonds, cook for an additional 5 minutes and remove from the heat. Garnish with the cilantro leaves."

    vegeterizer = Recipe_Vegeterizer()
    new_recipe, substitute_dict= vegeterizer.vegeterize(recipe_text)
    print(recipe_text)
    print(new_recipe)
    print(substitute_dict)