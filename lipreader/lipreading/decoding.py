"""CTC decoding and text metrics."""
import torch


def ctc_greedy_decode(logits, blank_id=0):
    """Greedy CTC decoder: collapse repeats and remove blanks."""
    pred_ids = torch.argmax(logits, dim=-1)  # (B, T)
    results = []
    for seq in pred_ids:
        out = []
        prev = None
        for t in seq.tolist():
            if t != blank_id and t != prev:
                out.append(t)
            prev = t
        results.append(out)
    return results


def ids_to_text(ids, num_to_char_dict):
    return "".join(num_to_char_dict.get(i, "?") for i in ids)


def compute_word_accuracy(pred_text, gt_text):
    """Compute word-level accuracy (forgiving to single char mismatches).
    
    Args:
        pred_text: Predicted text string
        gt_text: Ground truth text string
    
    Returns:
        word_accuracy: Fraction of words predicted correctly
        exact_match: Boolean, True if texts match exactly
    """
    pred_words = pred_text.lower().split()
    gt_words = gt_text.lower().split()
    
    if len(gt_words) == 0:
        return 1.0 if len(pred_words) == 0 else 0.0, len(pred_words) == 0
    
    correct = sum(1 for p, g in zip(pred_words, gt_words) if p == g)
    word_accuracy = correct / len(gt_words)
    exact_match = pred_text.lower() == gt_text.lower()
    
    return word_accuracy, exact_match
