#1
'''
To extract text from a PDF, you can use the PyPDF2 or pdfminer library in Python. Here's an example using PyPDF2:
'''

import PyPDF2

pdf_file = open('example.pdf', 'rb')
pdf_reader = PyPDF2.PdfFileReader(pdf_file)

text = ''
for page_num in range(pdf_reader.numPages):
    page = pdf_reader.getPage(page_num)
    text += page.extractText()

pdf_file.close()

'''
To identify different topics in the text, you can use a pre-trained GPT model like GPT-3 or a similar language model. 
You can use the Hugging Face Transformers library to load a pre-trained GPT model and generate topic labels for each sentence in the text. 
Here's an example using the GPT-2 model:
'''

from transformers import GPT2Tokenizer, GPT2Model

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

sentences = text.split('\n')
topic_dict = {}

for sentence in sentences:
    inputs = tokenizer.encode(sentence, return_tensors='pt')
    outputs = model(inputs)[0]
    topic = tokenizer.decode(outputs.argmax(dim=-1)[0])
    if topic not in topic_dict:
        topic_dict[topic] = [sentence]
    else:
        topic_dict[topic].append(sentence)
        
        
        
'''
This code will tokenize each sentence in the text, pass it through the GPT-2 model, and generate a topic label for the sentence. 
It will then categorize the sentences into a dictionary where the keys are the topics and the values are lists of sentences that belong to that topic. 
Note that this is a basic example, and you may need to adjust the code to fit your specific use case.
'''

#2
'''
To extract texts and categorize them into topics from a PDF file, you can use the PyPDF2 library in Python to read the text content of the PDF file, 
and then use the GPT model to classify the text into different categories.
'''

import PyPDF2
import openai
import pandas as pd

# Set up the OpenAI API key and model
openai.api_key = "YOUR_API_KEY"
model_engine = "davinci"  # Or use another GPT model if you prefer

# Load the PDF file
pdf_file = open('example.pdf', 'rb')
pdf_reader = PyPDF2.PdfFileReader(pdf_file)

# Extract the text content of each page
text_content = []
for i in range(pdf_reader.getNumPages()):
    page = pdf_reader.getPage(i)
    text_content.append(page.extractText())

# Classify each text content into topics using GPT
topics = []
for text in text_content:
    response = openai.Completion.create(
        engine=model_engine,
        prompt=text,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    topic = response.choices[0].text.strip()
    topics.append(topic)

# Create a pandas DataFrame to store the categorized texts
df = pd.DataFrame({'Text': text_content, 'Topic': topics})

# Export the DataFrame to a CSV file
df.to_csv('output.csv', index=False)


#3
'''
Here is an example Python code using the GPT model from Hugging Face's Transformers library that extracts text from a PDF file, 
categorizes the text into topics using topic modeling with Latent Dirichlet Allocation (LDA), and saves the topics and their corresponding 
text as a CSV file:
'''

import PyPDF2
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline

# Load GPT-2 model
gpt_model = pipeline('text-generation', model='gpt2')

# Open PDF file
pdf_file = open('example.pdf', 'rb')

# Read PDF file
pdf_reader = PyPDF2.PdfFileReader(pdf_file)
num_pages = pdf_reader.getNumPages()

# Extract text from PDF file
text = ""
for page_num in range(num_pages):
    page = pdf_reader.getPage(page_num)
    text += page.extractText()

# Tokenize and clean text
tokens = nltk.word_tokenize(text)
words = [word.lower() for word in tokens if word.isalpha()]

# Convert words to bag of words matrix
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(words)

# Run LDA topic modeling
lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
lda_model.fit(X)

# Get topic distribution for each document (in this case, only one document)
doc_topic_dist = lda_model.transform(X)

# Categorize text into topics
topic_names = ['Topic ' + str(i) for i in range(lda_model.n_components)]
topics = pd.DataFrame(doc_topic_dist, columns=topic_names)
topics['Text'] = pd.Series(words)
topic_cols = topics.columns.tolist()
topic_cols = topic_cols[-1:] + topic_cols[:-1]
topics = topics[topic_cols]

# Generate topic descriptions using GPT-2 model
topic_descriptions = []
for i in range(lda_model.n_components):
    topic_words = vectorizer.get_feature_names()[lda_model.components_[i].argsort()[-10:]]
    topic_words = ' '.join(topic_words)
    topic_description = gpt_model(f"Describe topic {i}: {topic_words}", max_length=100)[0]['generated_text'][24:].strip()
    topic_descriptions.append(topic_description)

# Save topics and their corresponding text as CSV file
topics.to_csv('topics.csv', index=False)

'''
This code uses the PyPDF2 library to extract text from a PDF file, the NLTK library to tokenize and clean the text, 
the scikit-learn library to convert the text into a bag of words matrix and run LDA topic modeling, and the Transformers 
library to generate topic descriptions using the GPT-2 model.

To use this code, replace 'example.pdf' with the path to your PDF file and adjust the number of topics (n_components) and maximum length of 
topic descriptions (max_length) as desired. The resulting CSV file will have a column for the text and one column for each topic, with values 
indicating the proportion of the text that belongs to each topic.
'''


#4

'''
Here's an example Python code using the PyPDF2 library to extract text from a PDF file and the GPT-2 model from the transformers library to 
categorize the text into different topics:
'''

import csv
import PyPDF2
import torch
from transformers import GPT2Tokenizer, GPT2Model

# Load the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# Define the function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    with open(file_path, mode='rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ''
        for page_num in range(reader.getNumPages()):
            page = reader.getPage(page_num)
            text += page.extractText()
        return text

# Define the function to categorize the text into different topics
def categorize_text(text):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_states = outputs.last_hidden_state
    topic_embeddings = last_hidden_states.mean(dim=1)
    # Replace the following with your own code to categorize the embeddings into different topics
    topics = ['Topic 1', 'Topic 2', 'Topic 3']
    topic_indices = [0, 1, 2, 0, 1, 2, 0, 1, 2] # Replace this with the actual topic indices for each sentence
    return topics, topic_indices

# Define the function to store the topics and their corresponding text as a CSV file
def store_as_csv(topics, topic_indices, text, output_file_path):
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Topic', 'Text'])
        for topic, topic_index, sentence in zip(topics, topic_indices, text.split('.')):
            writer.writerow([topic, sentence.strip()])

# Example usage
pdf_file_path = 'example.pdf'
output_file_path = 'output.csv'

text = extract_text_from_pdf(pdf_file_path)
topics, topic_indices = categorize_text(text)
store_as_csv(topics, topic_indices, text, output_file_path)

'''
Note that the categorize_text function currently just assigns a fixed set of topics to each sentence based on a fixed mapping from sentence index to 
topic index. You will need to replace this with your own code to categorize the embeddings into different topics based on your specific requirements.
'''

#5
'''
To extract text from a PDF file and categorize it into different topics, you can use the PyPDF2 and spaCy libraries in Python. 
You can then use a pre-trained GPT model from the Hugging Face Transformers library to identify the topics in the extracted text.

'''

import csv
import PyPDF2
import spacy
from transformers import pipeline

# Load the pre-trained GPT model
nlp = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Open the PDF file
with open("example.pdf", mode="rb") as pdf_file:
    # Read the contents of the PDF file
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    num_pages = pdf_reader.getNumPages()
    text = ""
    for page_num in range(num_pages):
        page_obj = pdf_reader.getPage(page_num)
        text += page_obj.extractText()

# Use spaCy to process the extracted text
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# Create a dictionary to store the topics and their corresponding text
topics = {}

# Loop through the sentences in the text and use the GPT model to classify each sentence
for sent in doc.sents:
    classification = nlp(sent.text)
    topic = classification.cats.get("LABEL_NAME")
    if topic:
        # If the topic is not already in the dictionary, add it
        if topic not in topics:
            topics[topic] = ""
        # Add the sentence to the text for the topic
        topics[topic] += sent.text

# Write the topics and their corresponding text to a CSV file
with open("output.csv", mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Topic", "Text"])
    for topic, text in topics.items():
        writer.writerow([topic, text])
        
        

#6
'''
To extract text from a PDF file in Python, you can use the PyPDF2 or pdfminer libraries. For topic modeling, we will use the Gensim library, 
which provides a fast and efficient implementation of the Latent Dirichlet Allocation (LDA) algorithm. 
'''
        
 
# Import libraries
import PyPDF2
import pdfminer
from gensim import corpora, models
import pandas as pd

# Open the PDF file
pdf_file = open('example.pdf', 'rb')

# Read the PDF file and extract the text
pdf_reader = PyPDF2.PdfFileReader(pdf_file)
text = ''
for i in range(pdf_reader.getNumPages()):
    page = pdf_reader.getPage(i)
    text += page.extractText()

# Preprocess the text
documents = text.split('\n\n')
texts = [[word for word in document.lower().split()] for document in documents]

# Create a dictionary and corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Train the LDA model
lda_model = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)

# Extract the topics and their corresponding texts
topics = []
for i in range(len(texts)):
    topic = lda_model.get_document_topics(corpus[i])
    topics.append(topic)

# Create a pandas dataframe to store the topics and their corresponding texts
df = pd.DataFrame(columns=['Topic', 'Text'])
for i in range(len(topics)):
    for j in range(len(topics[i])):
        df = df.append({'Topic': topics[i][j][0], 'Text': documents[i]}, ignore_index=True)

# Save the dataframe as a CSV file
df.to_csv('topics.csv', index=False)


#7
'''
o accomplish this task, we will use the Python programming language and the OpenAI GPT-3 model, along with some additional 
libraries such as PyPDF2 and spaCy. Here's an outline of the steps we'll take:

1. Install and import necessary libraries
2. Load the PDF file
3. Extract the text from the PDF
4. Preprocess the text data
5. Use the GPT-3 model to identify topics
6. Categorize the text into topics
7. Store the topics and their corresponding text as a dictionary
'''

# Step 1: Install and import necessary libraries
!pip install PyPDF2
!pip install spacy
!pip install openai

import PyPDF2
import spacy
import openai

# Step 2: Load the PDF file
pdf_file = open('example.pdf', 'rb')
pdf_reader = PyPDF2.PdfFileReader(pdf_file)

# Step 3: Extract the text from the PDF
pdf_text = ''
for page_num in range(pdf_reader.numPages):
    page_obj = pdf_reader.getPage(page_num)
    pdf_text += page_obj.extractText()

# Step 4: Preprocess the text data
nlp = spacy.load('en_core_web_sm')
doc = nlp(pdf_text)
sentences = [sent.string.strip() for sent in doc.sents]

# Step 5: Use the GPT-3 model to identify topics
openai.api_key = 'YOUR_API_KEY'
model_engine = 'text-davinci-002'
prompt = 'Please categorize the following texts into different topics:\n\n' + '\n'.join(sentences)
response = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.7,
)

# Step 6: Categorize the text into topics
topics = {}
for choice in response.choices:
    text = choice.text.strip()
    for line in text.split('\n'):
        if ':' in line:
            topic, sentence = line.split(':', 1)
            topic = topic.strip()
            sentence = sentence.strip()
            if topic not in topics:
                topics[topic] = []
            topics[topic].append(sentence)

# Step 7: Store the topics and their corresponding text as a dictionary
print(topics)
