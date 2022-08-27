def accuracy(predictions, gold):
    return (predictions.view(gold.size()) == gold).sum() / len(gold)
