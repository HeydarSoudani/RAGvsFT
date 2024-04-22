from difflib import SequenceMatcher

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard
import numpy as np
from nltk.metrics.distance import edit_distance


def seq_match_sim(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()

def bert_score_sim(s1, s2):
    candidate = word_tokenize(s1.lower())
    reference = word_tokenize(s2.lower())
    return sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25))

def cos_sim(s1, s2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([s1, s2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
    return cosine_sim

def jaccard_sim(s1, s2):
    set1 = set(s1.split())
    set2 = set(s2.split())
    jaccard_sim = 1 - jaccard(list(set1), list(set2))
    return jaccard_sim
    
def levenshtein_distance(s1, s2):
    edit_dist = edit_distance(s1, s2)
    return edit_dist

def vote(samples):
    # Initialize similarity scores dictionary
    similarity_scores = {sample: 0 for sample in samples}

    # Calculate similarity scores for each sample
    for i, si in enumerate(samples):
        for j, sj in enumerate(samples):
            if i != j:
                if similarity_method == 'seq_match':
                    similarity_scores[si] += seq_match_sim(si, sj) 
                
                elif similarity_method == 'bert_score':
                    similarity_scores[si] += bert_score_sim(si, sj)
                
                elif similarity_method == 'cosine':
                    similarity_scores[si] += cos_sim(si, sj)
                
                elif similarity_method == 'jaccard':
                    similarity_scores[si] += jaccard_sim(si, sj)
                
                elif similarity_method == 'levenshtein':
                    similarity_scores[si] += levenshtein_distance(si, sj)

    # Find the sample with the maximum similarity score
    final_answer = max(similarity_scores, key=similarity_scores.get)
    return final_answer

