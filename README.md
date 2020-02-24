# Vegetarize_It

Input: recipe
Output: same recipe but without meat products

Ideas:
- for training: vegetarian recipes -> recognize all vegetables/ meat replacements -> mask them -> try to recreate same recipe
- for using: mask all meat products in meat recipe -> input in same system
- masking: NER or gazeteers
- first stage: only delete spezific products
- second stage: also delete how to use them (fry, cook, etc)
