import json

import torch
from transformers import BertForMaskedLM, BertTokenizer


class MLMModel:

    def __init__(self):
        self.model: BertForMaskedLM = BertForMaskedLM.from_pretrained(
            pretrained_model_name_or_path='Foodbert/foodbert/data/mlm_output/checkpoint-final')
        with open('Foodbert/foodbert/data/used_ingredients.json', 'r') as f:
            used_ingredients = json.load(f)
        self.tokenizer = BertTokenizer(vocab_file='Foodbert/foodbert/data/bert-base-cased-vocab.txt', do_lower_case=False,
                                       max_len=128, never_split=used_ingredients)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

    def predict_substitutes(self, sentence, ingredient_name, with_masking=True):
        search_id = self.tokenizer.mask_token_id if with_masking else \
            self.tokenizer.convert_tokens_to_ids([ingredient_name])[0]
        sentence = sentence.replace('!', ' !').replace('?', ' ?').replace('.', ' .').replace(':', ' :').replace(',', ' ,')
        sentence = ' ' + sentence + ' '

        all_ordered_substitutes = []

        masked_sentence = sentence.replace(f' {ingredient_name} ', ' [MASK] ')
        input_ids = torch.tensor(self.tokenizer.encode(masked_sentence, add_special_tokens=True)).unsqueeze(0).to(device=self.device)
        prediction_scores = self.model(input_ids, masked_lm_labels=input_ids)[1][0]
        ingredient_scores = prediction_scores[input_ids[0] == search_id]

        for i in range(len(ingredient_scores)):
            ingredient_score = ingredient_scores[i]
            softmax_scores = ingredient_score.softmax(dim=0)
            indices = torch.sort(ingredient_score, descending=True).indices
            ordered_substitutes = self.tokenizer.convert_ids_to_tokens(indices)
            softmax_scores = softmax_scores[indices].tolist()
            all_ordered_substitutes.append((ordered_substitutes, softmax_scores))

        return all_ordered_substitutes
