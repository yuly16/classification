import numpy as np
import torch
def class_precision(labels, pred):
    # Calculates the precision for each class
    true_positives = np.zeros(20)
    predicted_positives = np.zeros(20)
    pred_c = torch.round(torch.clamp(labels * pred, 0, 1))
    for i in range(20):
        true_positives[i] = torch.sum(pred_c[:, i])
        predicted_positives[i] = torch.sum(torch.round(torch.clamp(pred[:, i], 0, 1)))
    class_precision = true_positives / predicted_positives
    return class_precision

def class_recall(labels, pred):
    # Calculates the recall for each class
    true_positives = np.zeros(20)
    possible_positives = np.zeros(20)
    pred_c = torch.round(torch.clamp(labels * pred, 0, 1))
    for i in range(20):
        true_positives[i] = torch.sum(pred_c[:, i])
        possible_positives[i] = torch.sum(labels[:, i])
    class_recall = true_positives / possible_positives
    
    return class_recall

def class_fbeta_score(labels, pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall for each class
	fbeta_score = np.zeros(20)
	if beta < 0:
		raise ValueError('Loewst is zero')
	# if torch.sum(torch.round(torch.clamp(labels[:, i], 0, 1))) == 0:
	# 	return fbeta_score
	p = class_precision(labels, pred)
	r = class_recall(labels, pred)
	bb = beta ** 2
	fbeta_score = (1 + bb) * (p * r) / (bb * p + r)
	return fbeta_score

# Definition of mAcc and wAcc

def mAcc(labels, pred):
    labels = np.array(labels.cpu())
    pred = np.array(pred.cpu())
    pred = np.where(pred > 0.5, 1, 0) # Threshold can be changed
    macc_per_label = np.zeros(20)
    bin_classify = np.where(labels == pred, 1, 0)
    for i in range(20):
        macc_per_label[i] = np.sum(bin_classify[:, i]) / len(labels)
    return np.sum(macc_per_label) / 20


def wAcc(labels, pred):
    labels = np.array(labels.cpu())
    pred = np.array(pred.cpu())
    pred = np.where(pred > 0.5, 1, 0)
    macc_per_label = np.zeros(20)
    weight = np.zeros(20)
    for i in range(20):
        weight[i] = np.sum(labels[:, i]) / len(labels)
    bin_classify = np.where(labels == pred, 1, 0)
    for i in range(20):
        macc_per_label[i] = np.sum(bin_classify[:, i]) / len(labels)
    return np.dot(macc_per_label, weight) / np.sum(weight)


def threshold(x):
    return 0.5 - x 
thre = 0.6
def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + 1e-6)
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clamp(y_true, 0, 1)))
    recall = true_positives / (possible_positives + 1e-6)
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    # Calculate the F score, the weighted harmonic mean of precision and recall
    if beta < 0:
        raise ValueError('Lowest is zero')

    # If there are no true positives fix the F score at 0 like sklearn
    if torch.sum(torch.round(torch.clamp(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + 1e-6)
    return fbeta_score


def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall
    return fbeta_score(y_true, y_pred, beta=1)
