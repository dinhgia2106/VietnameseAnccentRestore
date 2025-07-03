
def calculate_character_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Calculate character-level accuracy."""
    if len(predictions) != len(targets):
        return 0.0
    
    total_chars = 0
    correct_chars = 0
    
    for pred, target in zip(predictions, targets):
        total_chars += max(len(pred), len(target))
        
        # Character-by-character comparison
        for i in range(min(len(pred), len(target))):
            if pred[i] == target[i]:
                correct_chars += 1
    
    return correct_chars / total_chars if total_chars > 0 else 0.0

def calculate_word_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Calculate word-level exact match accuracy."""
    if len(predictions) != len(targets):
        return 0.0
    
    correct = sum(1 for pred, target in zip(predictions, targets) 
                  if pred.strip() == target.strip())
    
    return correct / len(predictions)

def calculate_bleu_score(predictions: List[str], targets: List[str]) -> float:
    """Calculate simplified BLEU score."""
    try:
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.tokenize import word_tokenize
        
        scores = []
        for pred, target in zip(predictions, targets):
            pred_tokens = list(pred)  # Character-level for Vietnamese
            target_tokens = [list(target)]
            
            score = sentence_bleu(target_tokens, pred_tokens)
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    except ImportError:
        # Fallback simple metric
        return calculate_character_accuracy(predictions, targets)
