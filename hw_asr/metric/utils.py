import editdistance


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 0 if len(predicted_text) == 0 else float('inf')
    distance = editdistance.distance(target_text, predicted_text)
    return distance / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    target_words = target_text.split(' ')
    predicted_words = predicted_text.split(' ')
    if len(target_words) == 0:
        return 0 if len(predicted_words) == 0 else float('inf')
    distance = editdistance.distance(target_words, predicted_words)
    return distance / len(target_words)
