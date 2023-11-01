import torch
from torch import nn
import transformers
from transformers import GPT2LMHeadModel
import os
from pathlib import Path

def build_model(config):
    
    if config['arch'] == 'gpt2':
        model = GPT2PromptTuning()

    return model

class GPT2PromptTuning(nn.Module):
    def __init__(self, soft_prompt_path=None, n_tokens=100, initialize_from_vocab=True, random_range=0.5):
        super().__init__()
        self.name = 'GPT2PromptTuning'
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Make sure to freeze Tranformers model
        for param in self.model.parameters():
            param.requires_grad = False

        if soft_prompt_path is not None:
            self.set_soft_prompt_embeds(soft_prompt_path)

        elif n_tokens is not None:
            print("Initializing soft prompt...")
            self.initialize_soft_prompt(
                n_tokens=n_tokens,
                initialize_from_vocab=initialize_from_vocab,
                random_range=random_range,
            )

    def set_soft_prompt_embeds(
        self,
        soft_prompt_path: str,
    ) -> None:
        """
        Args:
            soft_prompt_path: torch soft prompt file path
        """
        self.soft_prompt = torch.load(soft_prompt_path)
        self.n_tokens = self.soft_prompt.num_embeddings
        print(f"Set soft prompt! (n_tokens: {self.n_tokens})")

    def initialize_soft_prompt(
        self,
        n_tokens: int = 20,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
    ) -> None:
        self.n_tokens = n_tokens
        if initialize_from_vocab:
            init_prompt_value = self.model.transformer.wte.weight[:n_tokens].clone().detach()
        else:
            init_prompt_value = torch.FloatTensor(2, 10).uniform_(
                -random_range, random_range
            )
        self.soft_prompt = nn.Embedding(n_tokens, self.model.config.n_embd)
        # Initialize weight
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)
        print(f"Initialized soft prompt! (n_tokens: {self.n_tokens})")

    def _cat_learned_embedding_to_input(self, input_ids) -> torch.Tensor:
        inputs_embeds = self.model.transformer.wte(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds

    def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)

        n_batches = labels.shape[0]
        return torch.cat(
            [
                torch.full((n_batches, self.n_tokens), ignore_index).to(self.device),
                labels,
            ],
            dim=1,
        )

    def _extend_attention_mask(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.n_tokens), 1).to(self.device), attention_mask.to(self.device)],
            dim=1,
        )

    def save_soft_prompt(self, path: str, filename: str = "soft_prompt.model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.soft_prompt, os.path.join(path, filename))
        # print(f"Saved soft prompt: {os.path.join(path, filename)}")

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to(self.device)

        # if labels is not None:
        #     labels = self._extend_labels(labels).to(self.device)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask).to(self.device)
        # print(attention_mask.shape, inputs_embeds.shape, labels.shape)
        # Drop most of the args for now
        return self.model.forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
        )