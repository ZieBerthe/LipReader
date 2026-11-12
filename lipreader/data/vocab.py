"""Character vocabulary shared by the lip reader (CTC) and the TTS text encoder.

Index 0 is reserved for padding / the CTC blank.
"""

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!1234567890 "]
char_to_num = {char: idx + 1 for idx, char in enumerate(vocab)}
num_to_char = {idx: char for char, idx in char_to_num.items()}
num_to_char[0] = ''

NUM_CLASSES = len(vocab) + 1  # +1 for blank/padding at index 0


def encode_chars(chars):
    """Convert a sequence of characters to indices."""
    return [char_to_num[c] for c in chars]


def decode_indices(indices):
    """Convert indices back to a string (blank/padding decodes to '')."""
    return ''.join(num_to_char.get(i, '?') for i in indices)
