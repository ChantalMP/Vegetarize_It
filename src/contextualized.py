'''
Generates contextualized substitutes.
In the default configuration, it receives a recipe and an ingredient in the recipe, and replaces it with the most probably substitutes.
Recipe: recipe_text, Ingredient:ingredient
generate_substitute_in_text can directly be called with a text and ingredient to get N number of substitutes.
'''
import difflib
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from model import MLMModel
from Foodbert.normalisation.helpers.recipe_normalizer import RecipeNormalizer
from Foodbert.normalisation.normalize_recipe_instructions import normalize_instruction
from Foodbert.relation_extraction.prepare_dataset import split_reviews_to_sentences


def calculate_weight(key, ingredients_counts):
    if key == 'asparagus':
        key = 'asparagu'
    frequency = ingredients_counts['_'.join(key.split())]
    return 100000 / frequency


class ContextualizedSubstitutes:
    def __init__(self, meat_ingredients=[]):
        self.mlm_model = MLMModel()
        self.recipe_normalizer = RecipeNormalizer(lemmatization_types=['NOUN'])
        with open("Foodbert/data/cleaned_yummly_ingredients.json") as f:
            ingredients_yummly = json.load(f)
        self.ingredients_yummly_set = {tuple(ing.split(' ')) for ing in ingredients_yummly}
        substitutes_path = Path("Foodbert/foodbert_embeddings/data/substitutes_embeddings_high_recall.json")
        with substitutes_path.open() as f:
            self.all_predicted_subtitutes = {tuple(elem) for elem in json.load(f)}
        self.meat_ingredients = meat_ingredients

    def _clean_modified_string(self, original, modified):
        a = [' '] + original.split()
        b = [' '] + modified.split()
        final_string = ""
        matching_blocks = list(difflib.SequenceMatcher(None, a, b).get_matching_blocks())
        all_block_tuples = []
        for idx, matching_block in enumerate(matching_blocks):
            all_block_tuples.append(
                (b[matching_block.b:matching_block.b + matching_block.size], b[matching_block.b:matching_block.b + matching_block.size]))
            if idx < len(matching_blocks) - 1:
                next_match = matching_blocks[idx + 1]
                all_block_tuples.append(
                    (a[matching_block.a + matching_block.size:next_match.a], b[matching_block.b + matching_block.size:next_match.b]))

        for original, changed in all_block_tuples:
            if 'fish' in changed:
                final_string += ' '.join(changed) + ' '
            else:
                final_string += ' '.join(original) + ' '

        final_string = final_string.strip()
        return final_string

    def meat_free(self, ingr):
        for meat in self.meat_ingredients:
            if meat in ingr and not 'substitute' in ingr or ingr == 'vegetable':
                return False
        return True

    def _generate_substitutes_in_sentence(self, sentence):
        '''
        :return: list of lists (one per occurrence) of substitutes
        '''
        # normalize sentence and ingredient_name
        sentence = self.recipe_normalizer.model.pipe([sentence], n_process=1, batch_size=1)
        sentence = normalize_instruction(instruction_doc=next(sentence),
                                         yummly_ingredients_set=self.ingredients_yummly_set,
                                         instruction_normalizer=self.recipe_normalizer)

        # find all meats
        meats = []
        meats_with_substitutes = {}
        sentence_to_split = self.recipe_normalizer.model.pipe([sentence], n_process=1, batch_size=1)
        for word in next(sentence_to_split):
            for meat in self.meat_ingredients:
                if meat == word.norm_ and 'substitute' not in word.norm_:
                    meats.append(word.norm_)

        # find replacement for all meats
        for ingredient_to_replace in set(meats):
            ingredient_to_replace = self.recipe_normalizer.normalize_ingredients([ingredient_to_replace], strict=False)[0]
            ingredient_to_replace = '_'.join(ingredient_to_replace.split(' '))

            ordered_substitutes = self.mlm_model.predict_substitutes(sentence=sentence, ingredient_name=ingredient_to_replace,
                                                                 with_masking=True)

            ingredient_substitutes = {'_'.join(elem[1].split()) for elem in self.all_predicted_subtitutes if
                                  elem[0] == ' '.join(ingredient_to_replace.split('_'))}

            all_ordered_substitutes = []
            for occurence_substitutes in ordered_substitutes:
                occurence_substitutes_filtered = {ingr: score for ingr, score in zip(occurence_substitutes[0], occurence_substitutes[1]) if
                                      ingr in ingredient_substitutes and self.meat_free(ingr)}
                if len(occurence_substitutes_filtered) == 0:
                    occurence_substitutes_filtered = {ingr: score for ingr, score in zip(occurence_substitutes[0], occurence_substitutes[1]) if
                                             self.meat_free(ingr)}
                all_ordered_substitutes.append(occurence_substitutes_filtered)

            meats_with_substitutes[ingredient_to_replace] = all_ordered_substitutes

        return meats_with_substitutes

    def generate_substitute_in_text(self, text):
        with open('Foodbert/foodbert/data/ingredient_counts.json') as f:
            ingredients_counts = json.load(f)
            ingredients_counts = dict(ingredients_counts)
        sentences = split_reviews_to_sentences([text])
        predictions_per_meat = defaultdict(list)
        for sentence in sentences:
            substitutes_per_meat = self._generate_substitutes_in_sentence(sentence)
            for key in substitutes_per_meat:
                for prediction in substitutes_per_meat[key]:
                    predictions_per_meat[key].append(prediction)


        joined_predictions_per_meat = {}
        for key1 in predictions_per_meat:
            joined_substitutes = defaultdict(float)
            for prediction_dict in predictions_per_meat[key1]:
                for key2, value in prediction_dict.items():
                    joined_substitutes[key2] += value * calculate_weight(key, ingredients_counts)
            joined_substitutes = sorted(joined_substitutes, key=joined_substitutes.get, reverse=True)
            joined_predictions_per_meat[key1] = joined_substitutes

        return joined_predictions_per_meat

    def generate_substitutes_in_recipe(self, recipe_text, prefix="", suffix=""):
        substitutes_per_meat = self.generate_substitute_in_text(recipe_text)

        # normalize sentence and ingredient_name
        recipe_text = self.recipe_normalizer.model.pipe([recipe_text], n_process=-1, batch_size=1)
        recipe_text = normalize_instruction(instruction_doc=next(recipe_text),
                                            yummly_ingredients_set=self.ingredients_yummly_set,
                                            instruction_normalizer=self.recipe_normalizer)

        substitute_dict = {}
        for ingredient_to_replace, substitutes in substitutes_per_meat.items():
            substitute = substitutes[0]
            ingredient_to_replace = self.recipe_normalizer.normalize_ingredients([ingredient_to_replace], strict=False)[0]
            ingredient_to_replace = '_'.join(ingredient_to_replace.split(' '))

            # replace occurences
            recipe_text = recipe_text.replace('!', ' !').replace('?', ' ?').replace('.', ' .').replace(':', ' :').replace(',', ' ,')
            recipe_text = ' ' + recipe_text + ' '
            recipe_text = recipe_text.replace(f' {ingredient_to_replace} ', f' {prefix + substitute + suffix} ')
            recipe_text = recipe_text.replace(' !', '!').replace(' ?', '?').replace(' .', '.').replace(' :', ':').replace(' ,', ',')
            ingredient_to_replace = ingredient_to_replace.replace('_', ' ')
            substitute = substitute.replace('_', ' ')
            substitute_dict[ingredient_to_replace] = substitute

        recipe_text = recipe_text.replace('_', ' ')
        return recipe_text.strip(), substitute_dict


if __name__ == '__main__':
    with open('data/meats_normalized.txt') as f:
        meat_ingredients = f.read().splitlines()
    cs = ContextualizedSubstitutes(meat_ingredients=meat_ingredients)

    recipe_text = "Whisk together the yogurt, lemon juice, turmeric, garam masala and cumin in a large bowl. Put the chicken in, and coat with the marinade. Cover, and refrigerate (for up to a day). " \
                  "Add the chicken and marinade to the pan, and cook for 5 minutes, then add the chicken stock. " \
                  "Bring the mixture to a boil, then lower the heat and simmer, uncovered, for approximately 30 minutes. Stir in the cream and tomato paste, and simmer until the chicken is cooked through, approximately 10 to 15 minutes. " \
                  "Add the almonds, cook for an additional 5 minutes and remove from the heat. Garnish with the cilantro leaves."
    ingredient = 'chicken'

    new_recipe, substitute_dict = cs.generate_substitutes_in_recipe(recipe_text)
    #print(recipe_text)
    print(new_recipe)
    print(substitute_dict)