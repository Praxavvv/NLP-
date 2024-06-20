import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

# Ensure NLTK stopwords are downloaded
nltk.download('punkt')
nltk.download('stopwords')


def summarize_paragraph(paragraph):
    # Tokenize the paragraph into sentences
    sentences = sent_tokenize(paragraph)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))

    # Tokenize each sentence into words, remove stopwords, and lowercase
    words = [word.lower() for sentence in sentences for word in word_tokenize(sentence) if
             word.lower() not in stop_words]

    # Calculate word frequency
    word_frequency = FreqDist(words)

    # Score sentences based on average word frequency
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        score = sum(word_frequency[word.lower()] for word in word_tokenize(sentence) if word.lower() in word_frequency)
        sentence_scores[i] = score / len(word_tokenize(sentence)) if len(word_tokenize(sentence)) > 0 else 0

    # Get top N sentences based on scores (e.g., top 2 sentences)
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:2]

    # Generate the summarized paragraph
    summary = ' '.join(sentences[i] for i in sorted(top_sentences))

    return summary


# Example paragraph to summarize
example_paragraph = """
Natural language processing (NLP) is a subfield of artificial intelligence and linguistics. 
It involves the interaction between computers and humans through natural language. 
NLP is used to apply algorithms to identify and extract the natural language rules such as 
sentence structure, grammar, and word meanings. Applications of NLP can range from speech 
recognition, language translation, sentiment analysis, chatbot development, to text summarization. 
Techniques used in NLP include tokenization, stemming, lemmatization, and syntactic analysis. 
The development of NLP algorithms often involves machine learning and deep learning models 
trained on large datasets of text. NLTK (Natural Language Toolkit) is a popular library in Python 
for NLP tasks. It provides tools and resources like tokenizers, stopwords, word frequency counters, 
and algorithms for text processing and analysis. NLTK is widely used in academia and industry for 
research and development in natural language processing. In this paragraph, we explore the 
foundations of NLP and its practical applications, emphasizing its role in modern technology 
and its impact on various domains.
"""

# Call the summarize_paragraph function
summary = summarize_paragraph(example_paragraph)

# Print the summarized paragraph
print(summary)
