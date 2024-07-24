from typing import List, Union
from transformers import GPT2Tokenizer

class LlamaTokenizer:
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the LlamaTokenizer with a specified model.
        
        Args:
            model_name (str): The name of the model to use for tokenization.
                              Defaults to "gpt2".
        """
        self.tokenizer = self._load_tokenizer(model_name)

    @staticmethod
    def _load_tokenizer(model_name: str) -> GPT2Tokenizer:
        """
        Load the tokenizer for the specified model.
        
        Args:
            model_name (str): The name of the model to load the tokenizer for.
        
        Returns:
            GPT2Tokenizer: The loaded tokenizer.
        """
        return GPT2Tokenizer.from_pretrained(model_name)

    def count_tokens(self, text: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Count the number of tokens in the given text or list of texts.
        
        Args:
            text (Union[str, List[str]]): The text or list of texts to tokenize.
        
        Returns:
            Union[int, List[int]]: The number of tokens for each input text.
        """
        if isinstance(text, str):
            return len(self.tokenizer.encode(text))
        elif isinstance(text, list):
            return [len(self.tokenizer.encode(t)) for t in text]
        else:
            raise ValueError("Input must be a string or a list of strings.")

    def tokenize(self, text: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        """
        Tokenize the given text or list of texts.
        
        Args:
            text (Union[str, List[str]]): The text or list of texts to tokenize.
        
        Returns:
            Union[List[str], List[List[str]]]: The tokenized text(s).
        """
        if isinstance(text, str):
            return self.tokenizer.tokenize(text)
        elif isinstance(text, list):
            return [self.tokenizer.tokenize(t) for t in text]
        else:
            raise ValueError("Input must be a string or a list of strings.")
        
    def truncate_text(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)

def main():
    # Example usage
    tokenizer = LlamaTokenizer()
    
    sample_texts = [
        "Hello, this is a sample text to count tokens for Llama 3 70B model.",
        "This is another example sentence.",
        "And one more for good measure."
    ]
    
    # Count tokens
    token_counts = tokenizer.count_tokens(sample_texts)
    for text, count in zip(sample_texts, token_counts):
        print(f"Text: {text}")
        print(f"Token count: {count}")
        print()
    
    # Tokenize
    tokenized_texts = tokenizer.tokenize(sample_texts)
    for text, tokens in zip(sample_texts, tokenized_texts):
        print(f"Text: {text}")
        print(f"Tokens: {tokens}")
        print()

if __name__ == "__main__":
    main()