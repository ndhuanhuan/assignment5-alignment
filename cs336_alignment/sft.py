import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    assert len(prompt_strs) == len(output_strs)


    prompt_and_output_list = []
    for i, (prompt_str, output_str) in enumerate(zip(prompt_strs, output_strs, strict=True)):
        prompt_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt_str))
        output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output_str))
        prompt_and_output_list.append((prompt_ids, output_ids))

    batch_size = len(prompt_and_output_list)
    max_element = max(prompt_and_output_list, key = lambda x: len(x[0]) + len(x[1]))
    seq = len(max_element[0]) + len(max_element[1])

    input_ids = torch.zeros(batch_size, seq - 1, dtype=torch.int32)
    labels = torch.zeros(batch_size, seq - 1, dtype=torch.int32)
    response_mask = torch.zeros(batch_size, seq - 1, dtype=torch.bool)

    for i, prompt_and_output in enumerate(prompt_and_output_list):
        prompt_len = len(prompt_and_output[0])
        output_len = len(prompt_and_output[1])
        sequence = prompt_and_output[0] + prompt_and_output[1] + [tokenizer.pad_token_id] * (seq - prompt_len - output_len)
        input_ids[i, :seq-1] = torch.tensor(sequence[:-1])
        labels[i, :seq-1] = torch.tensor(sequence[1:])
        response_mask[i, prompt_len-1:prompt_len + output_len - 1 ] = True
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}