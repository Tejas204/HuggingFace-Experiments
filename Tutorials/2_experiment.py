from transformers import pipeline

# Zero shot classification
classifier = pipeline("zero-shot-classification")
res = classifier("This is a course about transformers library",
           candidate_labels=["education", "politics", "business"])

# Result
seq = res['sequence']
labels = res['labels']
scores = res['scores']

print(f"The sequence is: {seq}")
for label, score in zip(labels, scores):
    print(f"{label}: {score}")
