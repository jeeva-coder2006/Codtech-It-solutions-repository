
from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization")

# Example input
long_text = """
Artificial Intelligence (AI) is transforming industries by automating tasks, providing insights through data analysis, 
and improving decision-making processes. From healthcare to finance, AI applications are expanding rapidly. 
Modern AI systems use deep learning techniques to recognize patterns, classify data, and generate predictions. 
Natural Language Processing (NLP), a subset of AI, enables machines to understand and generate human language, 
powering chatbots, translators, and summarization tools.
"""

# Generate summary
summary = summarizer(long_text, max_length=50, min_length=25, do_sample=False)
print("Summary:", summary[0]['summary_text'])