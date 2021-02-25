import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

with open('data/dataset/test/es_test.labels', 'r', encoding="utf8") as file:
    true_labels = np.array([int(label.strip()) for label in file])

with open('results/lstm_ES.labels', 'r', encoding="utf8") as file:
    predicted_labels = np.array([int(label.strip()) for label in file])

cm = confusion_matrix(true_labels, predicted_labels)
us_labels = ['❤', '😍', '😂', '💕', '🔥', '😊', '😎', '✨', '💙',
             '😘', '📷', '🇺🇸', '☀', '💜', '😉', '💯', '😁', '🎄', '📸', '😜']
f = sns.heatmap(cm)
plt.show()
