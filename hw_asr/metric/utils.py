import editdistance


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return len(predicted_text)
    distance = editdistance.distance(target_text, predicted_text)
    return distance / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    target_words = target_text.split(' ')
    predicted_words = predicted_text.split(' ')
    if target_words == ['']:
        return 0 if predicted_words == [''] else len(predicted_words)
    distance = editdistance.distance(target_words, predicted_words)
    return distance / len(target_words)
