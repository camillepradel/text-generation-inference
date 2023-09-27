from enum import Enum
import string


class ResponseState(Enum):
    AFTER_NEWLINE = 1
    STARTING_ENTITY = 2
    WRITING_ENTITY = 3

class RestrictToListOfTextExtracts:
  """Forces generated output to be a numbered list of extracts from a given text"""
  def __init__(self, text, tokenizer):
  
    def get_candidates_starting_witk_token(token_ids, token_id_in_sequence, tokenizer, max_words=3):
      candidates = []

      def is_token_beginning_of_word(token_ids, token_id, tokenizer):
        token_str = tokenizer.convert_ids_to_tokens(token_ids[token_id])
        return token_str.startswith("▁") \
          or token_str.startswith("Ġ") or token_id==1 \
          or token_str[0] in string.punctuation

      if is_token_beginning_of_word(token_ids, token_id_in_sequence, tokenizer):
        candidate = []
        current_index = token_id_in_sequence
        words_count = 0
        while words_count < max_words:
          candidate.append(token_ids[current_index])        
          current_index += 1
          if current_index >= len(token_ids):
            break
          if is_token_beginning_of_word(token_ids, current_index, tokenizer):
            candidates.append(candidate)
            candidate = candidate.copy()
            words_count += 1
      return candidates

    def generate_candidates_tree(candidates):
      candidate_tree = {}
      def add_candidate_to_tree(candidate, candidate_tree):
        if len(candidate) == 0:
          candidate_tree["END"] = None
        else:
          first_token = candidate[0] # tokenizer.convert_ids_to_tokens(candidate[0])
          if first_token not in candidate_tree:
            candidate_tree[first_token] = {}
          add_candidate_to_tree(candidate[1:], candidate_tree[first_token])
      for candidate in candidates:
        add_candidate_to_tree(candidate, candidate_tree)
      return candidate_tree

    tokenized_text = tokenizer.encode(text)
    candidates = [
      get_candidates_starting_witk_token(tokenized_text, token_id_in_sequence, tokenizer, max_words=5)
      for token_id_in_sequence in range(len(tokenized_text))
    ]
    candidates = [candidate for candidates_inner in candidates for candidate in candidates_inner]
    self.candidate_tree = generate_candidates_tree(candidates)

    self.tokenizer = tokenizer
    self.state = None
    self.next_item = 1
    self.compulsory_next_tokens = []
    self.current_candidate_tree = None
    self.new_line_ids = [13] # hardcoded for Llama-2 family models

  def __call__(self, batch_id, input_ids):
    to_return = None
    if self.compulsory_next_tokens:
      to_return = [self.compulsory_next_tokens.pop(0)]
    else:
      last_input_id = int(input_ids[-1])
      if last_input_id in self.new_line_ids:
        self.state = ResponseState.AFTER_NEWLINE
      if self.state == ResponseState.AFTER_NEWLINE:
        to_return = self.tokenizer.convert_tokens_to_ids([str(self.next_item)])
        to_return.append(self.tokenizer.eos_token_id)
        self.next_item += 1
        self.compulsory_next_tokens = self.tokenizer.convert_tokens_to_ids(["."])
        self.state = ResponseState.STARTING_ENTITY
      elif self.state == ResponseState.STARTING_ENTITY:
        self.current_candidate_tree = self.candidate_tree
        self.state = ResponseState.WRITING_ENTITY
        to_return = list(self.current_candidate_tree.keys())
      elif self.state == ResponseState.WRITING_ENTITY:
        self.current_candidate_tree = self.current_candidate_tree[last_input_id]
        to_return = list(self.current_candidate_tree.keys())
        if "END" in to_return:
          to_return.remove("END")
          to_return.extend(self.new_line_ids)
          to_return.append(self.tokenizer.eos_token_id)
      else:
        to_return = self.new_line_ids + [self.tokenizer.eos_token_id]
    print(to_return, self.tokenizer.convert_ids_to_tokens(to_return))
    return to_return
