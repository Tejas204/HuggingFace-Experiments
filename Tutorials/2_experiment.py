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

    # messages_1 = [
#     {
#       "role": "user",
#       "content": [
#           {"type": "image", "path": "/Users/tejasdhopavkar/fiftyone/coco-2017/validation/data/000000055528.jpg"},
#           {"type": "text", "text": "Generate a caption for this image. Caption should be a one liner but descriptive enough"},
#         ],
#     },
# ]

# messages_2 = [
#     {
#       "role": "user",
#       "content": [
#           {"type": "image", "path": "/Users/tejasdhopavkar/fiftyone/coco-2017/validation/data/000000018491.jpg"},
#           {"type": "text", "text": "Generate a caption for this image. Caption should be a one liner but descriptive enough"},
#         ],
#     },
# ]
