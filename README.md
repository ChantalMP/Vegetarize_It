# Vegetarize_It

Input: recipe
Output: same recipe but without meat products

Ideas:
- for training: vegetarian recipes -> recognize all vegetables/ meat replacements -> mask them -> try to recreate same recipe
- for using: mask all meat products in meat recipe -> input in same system
- masking: NER or gazeteers
- first stage: only delete spezific products
- second stage: also delete how to use them (fry, cook, etc)


Recipe dataset: Recipe1M+: A Dataset for Learning Cross-Modal Embeddings for Cooking Recipes and Food Images. Marin, Javier and Biswas, Aritro and Ofli, Ferda and Hynes, Nicholas and Salvador, Amaia and Aytar, Yusuf and Weber, Ingmar and Torralba, Antonio, 2019 (http://pic2recipe.csail.mit.edu/)
