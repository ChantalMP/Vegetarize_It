from Foodbert.normalisation.helpers.recipe_normalizer import RecipeNormalizer

if __name__ == '__main__':
    recipe_normalizer = RecipeNormalizer(lemmatization_types=['NOUN'])
    with open('data/meats.txt') as f:
        meat_ingredients = f.read().splitlines()

    meat_ingredients_normalized = recipe_normalizer.normalize_ingredients(meat_ingredients, strict=False)
    for idx, elem in enumerate(meat_ingredients_normalized):
        meat_ingredients_normalized[idx] = '_'.join(elem.split(' '))

    with open('data/meats_normalized.txt', 'w') as f:
        for element in meat_ingredients_normalized:
            f.write(element + "\n")