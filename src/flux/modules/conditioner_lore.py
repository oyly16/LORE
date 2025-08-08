from torch import Tensor, nn
from transformers import (CLIPTextModel, CLIPTokenizer, T5EncoderModel,
                          T5Tokenizer)


class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, is_clip, **hf_kwargs):
        super().__init__()
        self.is_clip = is_clip
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        if not self.is_clip:
            pass

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]

    def forward_length(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        if not self.is_clip:
            pass

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        # -1 to delete the end token
        return outputs[self.output_key],batch_encoding['length']-1

    def get_word_embed(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=16,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = batch_encoding["input_ids"].to(self.hf_module.device)
        attention_mask = batch_encoding["attention_mask"].to(self.hf_module.device)
        
        outputs = self.hf_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        
        token_embeddings = outputs[self.output_key]  # [B, T, D]
        mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
        summed = (token_embeddings * mask).sum(dim=1)           # [B, D]
        counts = mask.sum(dim=1).clamp(min=1e-6)                 
        mean_pooled = summed / counts                            # [B, D]
        
        return mean_pooled


    def get_text_embeddings_with_diff(self, src_text: str, tgt_text: str, replacements: list[tuple[str, str, int, int]], show_tokens=False, return_embeds=False):
        batch_encoding = self.tokenizer(
            [src_text, tgt_text],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length",
        )
        
        src_ids, tgt_ids = batch_encoding["input_ids"]
        
        src_tokens = self.tokenizer.tokenize(src_text)
        tgt_tokens = self.tokenizer.tokenize(tgt_text)
        if show_tokens:
            print("src tokens", src_tokens)
            print("tgt tokens", tgt_tokens)
        
        src_dif_ids = []
        tgt_dif_ids = []
        def find_mappings(tokens,words,start_idx):
            if (words is None) or start_idx<0: # some samples do not need this
                return [-1]
            res = []
            flag = 0
            for i in range(start_idx,len(tokens)):
                this_token = tokens[i].strip('▁')
                if this_token == "":
                    continue
                if words.startswith(this_token):
                    res.append(i)
                    flag = 1
                    if words.endswith(this_token):
                        break
                    else:
                        continue
                if flag and words.endswith(this_token):
                    res.append(i)
                    break
                if flag:
                    res.append(i)
            return res
        
        for src_words, tgt_words, src_index, tgt_index in replacements:
            if src_words:
                src_dif_ids.append(find_mappings(src_tokens,src_words,src_index))
            else:
                src_dif_ids.append([-1])
            if tgt_words:
                tgt_dif_ids.append(find_mappings(tgt_tokens,tgt_words,tgt_index))
            else:
                tgt_dif_ids.append([-1])
            
        if return_embeds:
            outputs = self.hf_module(
                input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
                attention_mask=None,
                output_hidden_states=False,
            )
            embeddings = outputs[self.output_key]
        else:
            embeddings = (None,None)
        return embeddings[0], embeddings[1], src_dif_ids, tgt_dif_ids


