import os
import numpy as np
import random
import time

random.seed(time.process_time())

emojis = {0: '❤️', 1: '😍', 2: '😂', 3: '💕', 4: '🔥', 5: '😊', 6: '😎', 7: '✨', 8: '💙', 9: '😘',
          10: '📷', 11: '🇺🇸', 12: '☀', 13: '💜', 14: '😉', 15: '💯', 16: '😁', 17: '🎄', 18: '📸', 19: '😜'}

with open('data/dataset/test/us_test.text', 'r', encoding="utf8") as file:
    tweets = file.readlines()

with open('data/dataset/test/us_test.labels', 'r', encoding="utf8") as file:
    actual_labels = np.array([int(label.strip()) for label in file])

with open('results\lstm_US.labels', 'r', encoding="utf8") as file:
    predicted_labels = np.array([int(label.strip()) for label in file])

for i in range(0, 5):
    index = random.randint(0, len(predicted_labels))

    pred_label = predicted_labels[index]
    actual_label = actual_labels[index]
    tweet = tweets[index].rstrip()

    print("PREDICTION " + str(i))
    print("The Tweet is: " + tweet)
    print("The Predicted Emoji is: " + emojis[pred_label])
    print("The Actual Emoji is: " + emojis[actual_label])
    print("")
