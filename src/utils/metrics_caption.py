import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.spice.spice import Spice
from rouge import Rouge
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_captions(ground_truth, predicted):
    """
    Computes multiple evaluation metrics for predicted captions against ground truth captions.

    Args:
        ground_truth (list): A list of lists, where each inner list contains a single ground truth caption.
        predicted (list): A list of lists, where each inner list contains a single predicted caption.

    Returns:
        A dictionary containing the evaluation metrics.
    """
    # Tokenize the captions
    ground_truth_b = [[word_tokenize(caption)] for caption in ground_truth]
    predicted_b = [word_tokenize(caption) for caption in predicted]
    # Compute the BLEU score
    bleu_score = 0
    for i in range(len(predicted)):
        bleu_score += sentence_bleu(ground_truth_b[i], predicted_b[i])
    bleu_score = bleu_score / len(predicted)

    meteor_score_ = 0
    for i in range(len(predicted)):
        meteor_score_ += meteor_score(ground_truth_b[i], predicted_b[i])
    meteor_score_ = meteor_score_ / len(predicted)

    rouge = Rouge()
    hyps, refs = map(
        list, zip(*[[predicted[i], ground_truth[i]] for i in range(len(predicted))])
    )
    rouge_scores = rouge.get_scores(hyps, refs, avg=True)["rouge-l"]["f"]

    spice = Spice()
    hyps = {}
    refs = {}
    for i in range(len(predicted)):
        hyps[i] = [predicted[i]]
        refs[i] = [ground_truth[i]]
    spice_score, _ = spice.compute_score(hyps, refs)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # batch size of 16
    embeddings_gt = model.encode(ground_truth, batch_size=16)
    embeddings_gt = np.array(embeddings_gt)
    embeddings_pred = model.encode(predicted, batch_size=16)
    embeddings_pred = np.array(embeddings_pred)
    cosine_scores = []
    for i in range(len(predicted)):
        cosine_scores.append(
            cosine_similarity(
                embeddings_gt[i].reshape(1, -1), embeddings_pred[i].reshape(1, -1)
            )[0][0]
        )
    cosine_scores = np.array(cosine_scores)
    cosine_scores = np.mean(cosine_scores)
    # Return the evaluation metrics as a dictionary
    return {
        "BLEU": bleu_score,
        "METEOR": meteor_score_,
        "ROUGE": rouge_scores,
        "sent-sim": cosine_scores,
        "SPICE": spice_score,
    }


def test_captions():
    # Define the ground truth captions and predicted captions
    ground_truth = ["A cat is sitting on a table.", "A person is riding a bike."]
    predicted = ["A cat is lying on a table.", "A person is biking."]

    # Evaluate the predicted captions against the ground truth captions
    metrics = evaluate_captions(ground_truth, predicted)

    # Print the evaluation metrics
    for metric_name, metric_value in metrics.items():
        print("{}: {:.2f}".format(metric_name, metric_value))
