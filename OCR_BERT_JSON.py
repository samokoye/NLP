'''
produce two sentences similar to the following medical note below:
“Mr brown was admitted to our hospital, the lady of the lake medical 
center on the 3rd of March 2023. Brown has a heart disease and chronic back pain. 
After receiving professional treatment, bed rest was prescribed to him”
'''
#The result was:
'''
"Ms. Smith was admitted to St. Joseph's Hospital on March 10th, 2023, with a diagnosis 
of pneumonia and asthma. Following medical intervention, Ms. Smith was advised to undergo 
a period of rest and recovery."

"Mr. Johnson presented to the emergency department 
at Mercy Hospital on March 15th, 2023, with symptoms 
of a stroke. A comprehensive medical examination was 
conducted, and it was determined that Mr. Johnson required 
rest and rehabilitation to aid in his recovery."
'''

"""
here's an example code using the spaCy library to perform
 medical named entity recognition on the sentences generated
 by the previous code and output the results in a JSON format:
"""
import spacy
import random
import json

# load the spaCy model for medical named entity recognition
nlp = spacy.load("en_ner_bc5cdr_md")

patient_names = ["Ms. Johnson", "Mr. Rodriguez", "Mrs. Smith", "Mr. Lee", "Ms. Nguyen", "Mr. Patel", "Mrs. Garcia", "Mr. Kim", "Ms. Williams", "Mr. Brown"]
hospital_names = ["Memorial Hospital Center", "St. Francis Medical Center", "Central Hospital", "Riverside Medical Center", "Green Valley Hospital", "Fairview Hospital", "Pineview Medical Center", "Sunnybrook Hospital", "Oakdale Regional Medical Center", "Parkside Medical Center"]
conditions_list = ["hypertension and shortness of breath", "heart attack, diabetes, and high blood pressure", "cancer and chronic pain", "asthma and allergies", "pneumonia and bronchitis", "depression and anxiety", "fractured arm and concussion", "migraines and seizures", "kidney stones and UTI", "rheumatoid arthritis and fibromyalgia"]
treatment_list = ["rest and avoidance of strenuous activity", "treatment, rest, and adherence to a strict diet and exercise regimen", "chemotherapy and radiation therapy", "inhaled medications and nebulizer treatments", "antibiotics and breathing treatments", "counseling and therapy sessions", "immobilization and physical therapy", "medications and lifestyle modifications", "pain management and surgery", "medications and physical therapy"]

results = []

for i in range(100):
    patient_name = random.choice(patient_names)
    hospital_name = random.choice(hospital_names)
    conditions = random.choice(conditions_list)
    treatment = random.choice(treatment_list)
    admission_date = f"{random.randint(1, 31)}-{random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])}-{random.randint(2022, 2023)}"
    
    sentence1 = f"{patient_name} was admitted to {hospital_name} on {admission_date}. {patient_name} has a history of {conditions}."
    sentence2 = f"Following medical evaluation and treatment, {patient_name} has been advised to {treatment} for a period of time."
    
    # perform medical named entity recognition on both sentences
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)
    
    # extract named entities and their labels
    entities1 = [{"text": ent.text, "label": ent.label_} for ent in doc1.ents]
    entities2 = [{"text": ent.text, "label": ent.label_} for ent in doc2.ents]
    
    # combine the named entities from both sentences
    entities = entities1 + entities2
    
    # create a dictionary with the sentence and its named entities
    result = {"sentence1": sentence1, "sentence2": sentence2, "entities": entities}
    
    # add the dictionary to the list of results
    results.append(result)

# write the results to a JSON file
with open("medical_ner_results.json", "w") as f:
    json.dump(results, f, indent=2)

#This code uses the spaCy model en_ner_bc5cdr_md to 
#perform medical named entity

##########
'''
here are three medical notes with different sentence structures:

"Following a thorough evaluation, 
Mr. Johnson was admitted to our medical facility on March 25th, 2023, 
with a diagnosis of pneumonia and hypertension. After administering 
appropriate treatment, we recommended a brief period of observation to 
monitor his progress."
"Ms. Garcia arrived at our hospital on April 10th, 2023, 
with a history of diabetes and hypertension. 
After conducting a series of tests and evaluations, 
we determined that surgery was necessary. 
Following the procedure, we initiated a personalized 
care plan to facilitate her recovery."
"On May 2nd, 2023, Mr. Patel was transferred 
to our facility with a diagnosis of acute myocardial infarction. 
We immediately began administering treatment and monitoring his condition. 
After several days of observation, we recommended a course of physical therapy 
to aid in his recovery."
'''
import random

diagnoses = [('pneumonia', 'hypertension'),
             ('diabetes', 'hypertension'),
             ('acute myocardial infarction',)]

admissions = [('March 25th, 2023', 'Mr. Johnson'),
              ('April 10th, 2023', 'Ms. Garcia'),
              ('May 2nd, 2023', 'Mr. Patel')]

treatments = [('observation', 'progress'),
              ('surgery', 'recovery'),
              ('physical therapy', 'recovery')]

sentences = []
for i in range(3):
    diagnosis = random.choice(diagnoses)
    admission = random.choice(admissions)
    treatment = random.choice(treatments)
    
    sentence = f"On {admission[0]}, {admission[1]} was admitted to our medical facility with a diagnosis of {diagnosis[0]} and {diagnosis[1]}. After administering appropriate treatment, we recommended a brief period of {treatment[0]} to monitor {treatment[1]}."
    sentences.append(sentence)

print(sentences)
##########
"""
here's the Python code to produce 500
 medical notes with unique sentence 
structures and illnesses, and save the 
results in both JSON and Pandas DataFrame 
formats:
"""
import random
import pandas as pd
import json

diagnoses = [('pneumonia', 'hypertension'),
             ('diabetes', 'hypertension'),
             ('acute myocardial infarction',),
             ('cancer', 'heart disease'),
             ('asthma', 'chronic obstructive pulmonary disease'),
             ('malaria',),
             ('gastrointestinal infection', 'food poisoning'),
             ('kidney stones',)]

admissions = [('March 25th, 2023', 'Mr. Johnson'),
              ('April 10th, 2023', 'Ms. Garcia'),
              ('May 2nd, 2023', 'Mr. Patel'),
              ('June 5th, 2023', 'Ms. Lee'),
              ('July 15th, 2023', 'Mr. Rodriguez'),
              ('August 1st, 2023', 'Ms. Singh'),
              ('September 18th, 2023', 'Mr. Chen'),
              ('October 21st, 2023', 'Ms. Kim'),
              ('November 11th, 2023', 'Mr. Davis'),
              ('December 23rd, 2023', 'Ms. Hernandez')]

treatments = [('observation', 'progress'),
              ('surgery', 'recovery'),
              ('physical therapy', 'recovery'),
              ('medication', 'symptoms'),
              ('radiation therapy', 'tumor'),
              ('chemotherapy', 'cancer'),
              ('dietary changes', 'digestive health'),
              ('rest', 'fatigue'),
              ('oxygen therapy', 'breathing')]

sentences = []
for i in range(500):
    diagnosis = random.choice(diagnoses)
    admission = random.choice(admissions)
    treatment = random.choice(treatments)
    
    sentence_structure = random.choice(range(1, 4))
    if sentence_structure == 1:
        sentence = f"On {admission[0]}, {admission[1]} was admitted to our medical facility with a diagnosis of {diagnosis[0]} and {diagnosis[1]}. After administering appropriate treatment, we recommended a brief period of {treatment[0]} to monitor {treatment[1]}."
    elif sentence_structure == 2:
        sentence = f"{admission[1]} was admitted to our hospital on {admission[0]} with {diagnosis[0]} and {diagnosis[1]} diagnoses. Following a series of tests, {treatment[0]} was recommended to manage {treatment[1]}."
    else:
        sentence = f"With a diagnosis of {diagnosis[0]} and {diagnosis[1]}, {admission[1]} arrived at our facility on {admission[0]}. After receiving professional treatment, we prescribed a course of {treatment[0]} to aid in {treatment[1]} management."
    sentences.append(sentence)

# Save the sentences in JSON format
with open('medical_notes.json', 'w') as f:
    json.dump(sentences, f)

# Save the sentences in a Pandas DataFrame format
df = pd.DataFrame({'Medical Notes': sentences})
df.to_csv('medical_notes.csv', index=False)
######

"""
here's an updated Python code that generates 
500 medical notes with unique doctor's name 
and hospital stay duration:
"""
import random
import json

doctors = ['Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown', 'Dr. Davis']
illnesses = ['diabetes', 'migraine', 'pneumonia', 'asthma', 'cancer']
verbs = ['was diagnosed with', 'is suffering from', 'has been admitted with', 'is being treated for', 'is experiencing']
admissions = ['admitted to', 'checked into', 'brought to', 'transferred to', 'referred to']
hospitals = ['St. Mary\'s Hospital', 'University Hospital', 'Johns Hopkins Hospital', 'Mayo Clinic', 'Massachusetts General Hospital']
durations = ['for a week', 'for 10 days', 'for two weeks', 'for a month', 'for several months']

notes = []

for i in range(500):
    doctor = random.choice(doctors)
    illness = random.choice(illnesses)
    verb = random.choice(verbs)
    admission = random.choice(admissions)
    hospital = random.choice(hospitals)
    duration = random.choice(durations)
    
    sentence_structure = random.randint(1, 4) # choose a random sentence structure
    if sentence_structure == 1:
        sentence = f"{doctor} admitted the patient with {illness} to {hospital} and prescribed bed rest for {duration}."
    elif sentence_structure == 2:
        sentence = f"{doctor} diagnosed the patient with {illness} and referred them to {hospital} for further treatment. The patient stayed in the hospital for {duration}."
    elif sentence_structure == 3:
        sentence = f"The patient was {admission} {hospital} and {doctor} treated them for {illness}. The hospital stay lasted {duration}."
    else:
        sentence = f"{doctor} is currently treating the patient for {illness} at {hospital} and expects a hospital stay of {duration}."
    
    note = {
        'doctor': doctor,
        'sentence': sentence
    }
    
    notes.append(note)

# save result in json file
with open('medical_notes.json', 'w') as f:
    json.dump(notes, f)

# convert to dataframe (optional)
import pandas as pd

df = pd.DataFrame(notes)
print(df.head())

######

"""

"""



'''
Generating lots of medical data based on just one sample medical data is a challenging task, but it is possible with the 
help of machine learning techniques such as data augmentation.

Data augmentation is a technique in which the existing data is augmented by applying various transformations to the data, 
such as flipping, rotating, scaling, and adding noise. These transformations can create new variations of the existing data, 
which can then be used for training machine learning models.

To generate lots of medical data from a single sample medical data, you can use data augmentation techniques such as rotating the 
image at different angles, flipping the image horizontally and vertically, adding noise to the image, and changing the contrast and 
brightness of the image. You can also use other techniques such as cropping and resizing the image.

There are many open-source libraries available in Python such as TensorFlow and Keras that can be used to implement data augmentation 
techniques for medical images. By using these techniques, you can generate a large dataset of medical images that can be used to train 
machine learning models for various medical applications.
'''
--================--==============--============
'''
To build a machine learning model that can read the text inside a medical document image, create synthetic data and structure the information 
into sections in a JSON format, we can follow the following steps:

Step 1: Data collection and preprocessing

Collect a dataset of medical documents that contain unstructured patient illness information in image format.
Preprocess the images by resizing, converting to grayscale, and applying filters to improve the image quality.
Step 2: Text extraction

Use Optical Character Recognition (OCR) techniques to extract the text from the image.
Clean the extracted text by removing noise, punctuation, and special characters.
Step 3: Text classification

Train a machine learning model to classify the extracted text into different sections such as patient information, symptoms, diagnosis, medications, etc.
Use techniques such as Named Entity Recognition (NER) to identify important entities such as drug names, dosages, and frequency.
Step 4: Synthetic data generation

Generate synthetic data by augmenting the existing data using techniques such as data augmentation, text generation, and image processing.
This will help improve the model's performance by increasing the size of the dataset and exposing it to different types of images and text.
Step 5: Structuring information into JSON format

Once the text has been classified into different sections, the information can be structured into a JSON format.
Create a template for the JSON format with different sections such as patient information, symptoms, diagnosis, medications, etc.
Populate the JSON template with the relevant information extracted from the image.
Step 6: Model evaluation and refinement

Evaluate the model's performance on a test set using metrics such as precision, recall, and F1-score.
Refine the model by fine-tuning the hyperparameters and adjusting the architecture as necessary.
Once the model is trained and evaluated, it can be used to extract the text from medical documents, create synthetic data, and structure the information 
into a JSON format automatically. This can help save time and improve the accuracy of the information extracted from medical documents.
'''
--================--============
'''
1. Data pre-processing and image extraction
The first step would be to extract the text from the image. This can be done using Optical Character Recognition (OCR) techniques. Here's some sample code to extract the text from an image using the Tesseract OCR engine:
python
'''
import pytesseract
from PIL import Image

# Load image
image = Image.open('medical_document.jpg')

# Extract text using OCR
text = pytesseract.image_to_string(image)


'''
Text cleaning and pre-processing
The extracted text may contain noise and unwanted characters. We need to clean and preprocess the text before feeding it to our machine learning model. 
Here's some sample code to clean the text:
'''
import re

# Remove unwanted characters and symbols
text = re.sub('[^a-zA-Z0-9\n\.]', ' ', text)

# Remove excess whitespace
text = ' '.join(text.split())

#OR
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Remove special characters from the text
text = re.sub(r'\W', ' ', text)

# Convert the text to lowercase
text = text.lower()

# Tokenize the text
tokens = word_tokenize(text)

# Remove stop words from the tokens
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if not word in stop_words]

# Lemmatize the tokens
lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(word) for word in tokens]

# Join the tokens back into a string
text = ' '.join(tokens)



'''
3. Named Entity Recognition (NER)
The next step is to extract relevant entities such as names, locations, and dates from the text. This can be done using Named Entity Recognition (NER) 
techniques. Here's some sample code to extract named entities using the spaCy library:
'''
import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Process the text with spaCy
doc = nlp(text)

# Extract named entities
entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]

#OR

import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Perform semantic analysis on the text
doc = nlp(text)

# Extract the entities from the analyzed text
entities = []
for ent in doc.ents:
    entities.append((ent.text, ent.label_))

'''
4. Synthesizing Data
Now that we have extracted the relevant entities, we can generate synthetic data based on this. Here's some sample code to generate synthetic 
data using the Faker library:
'''
from faker import Faker

# Initialize Faker with a seed value for reproducibility
fake = Faker(seed=123)

# Generate synthetic data based on the extracted entities
data = {}
for entity in entities:
    if entity['label'] == 'PERSON':
        data['patient_name'] = entity['text']
        data['patient_address'] = fake.address()
        data['patient_email'] = fake.email()
    elif entity['label'] == 'DATE':
        data['date'] = entity['text']
    elif entity['label'] == 'DISEASE':
        data['disease'] = entity['text']
    elif entity['label'] == 'MEDICATION':
        data['medication'] = entity['text']

#OR

# Create synthetic data from the extracted entities
data = {}
for entity in entities:
    if entity[1] in data:
        data[entity[1]].append(entity[0])
    else:
        data[entity[1]] = [entity[0]]
        
'''
5. Structuring Data into JSON Format
Finally, we need to structure the extracted and synthesized data into a JSON format. Here's some sample code to do this:
'''
import json

# Structure the data into a JSON format
json_data = json.dumps({
    'patient': {
        'name': data.get('patient_name', ''),
        'address': data.get('patient_address', ''),
        'email': data.get('patient_email', ''),
    },
    'date': data.get('date', ''),
    'disease': data.get('disease', ''),
    'medication': data.get('medication', ''),
})

# Print the JSON data
print(json_data)


--=======---===============--====================

#SAMPLE
#train a NER model using spaCy
import spacy
from spacy.tokens import Doc, Span, Token

# Load a blank spaCy model
nlp = spacy.blank("en")

# Define the entity labels
LABELS = ["PATIENT_NAME", "ILLNESS", "SYMPTOMS", "MEDICATION", "DOSAGE"]

# Define the training data
TRAIN_DATA = [
    ("John Smith has a fever", {"entities": [(0, 10, "PATIENT_NAME"), (20, 25, "ILLNESS"), (26, 31, "SYMPTOMS")]}),
    ("Mary Johnson is taking aspirin", {"entities": [(0, 12, "PATIENT_NAME"), (21, 28, "MEDICATION")]}),
    ("Bob Miller is allergic to peanuts", {"entities": [(0, 10, "PATIENT_NAME"), (20, 27, "SYMPTOMS")]}),
    ("Jane Brown is prescribed 500mg of antibiotics", {"entities": [(0, 9, "PATIENT_NAME"), (23, 35, "MEDICATION"), (36, 40, "DOSAGE")]}),
]

# Add the entity recognizer to the pipeline
ner = nlp.create_pipe("ner")
nlp.add_pipe(ner)

# Add the labels to the entity recognizer
for label in LABELS:
    ner.add_label(label)

# Train the NER model
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])
    example = Doc(nlp.vocab, words=_.split())
    example.ents = [Span(example, start=ent[0], end=ent[1], label=ent[2]) for ent in annotations.get("entities")]
    nlp.update([example], [annotations])

--==========---=============---=======--------===================
        +-------------------+
        |  Medical Text in  |
        |  Image (Optional) |
        +-------------------+
                   |
                   v
        +-------------------+
        |   OCR (Optional)  |
        +-------------------+
                   |
                   v
        +-------------------+
        |    NLP Pipeline   |
        +-------------------+
                   |
                   v
        +-------------------+
        |    Tokenization   |
        +-------------------+
                   |
                   v
        +-------------------+
        |   Stopword Removal|
        +-------------------+
                   |
                   v
        +-------------------+
        |  Part-of-Speech   |
        |       (POS)       |
        +-------------------+
                   |
                   v
        +-------------------+
        | Named Entity Rec. |
        |       (NER)       |
        +-------------------+
                   |
                   v
        +-------------------+
        |   Dependency      |
        |       Parsing     |
        +-------------------+
                   |
                   v
        +-------------------+
        |   Structured Data |
        +-------------------+

--=========----=======----========---======
import cv2
import pytesseract
import re
import spacy
import transformers
import json

# Preprocess and clean the data
def preprocess_medical_data(medical_data):
    # Remove any irrelevant information from the medical data
    # (e.g., headers, footers, boilerplate text)
    # Standardize the format of the medical notes
    # (e.g., remove line breaks, convert to lowercase)
    # Return the cleaned medical data
    pass

# Train a language model
def train_language_model(medical_data):
    # Preprocess the medical data
    cleaned_data = preprocess_medical_data(medical_data)
    
    # Train a language model, such as BERT, on the cleaned data
    # (e.g., using the transformers library)
    model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased')
    
    # Return the trained model
    return model

# Add an OCR model to the pipeline
def ocr_process_medical_images(medical_images):
    # Enhance the medical images using image processing techniques
    # (e.g., noise reduction, contrast enhancement)
    enhanced_images = []
    for image in medical_images:
        enhanced_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        enhanced_images.append(enhanced_image)
    
    # Use an OCR model, such as pytesseract, to read the medical text from the images
    # (e.g., using the OCR-A font for improved accuracy)
    text_outputs = []
    for enhanced_image in enhanced_images:
        text_output = pytesseract.image_to_string(enhanced_image, lang='ocr_a')
        text_outputs.append(text_output)
    
    # Return the OCR outputs
    return text_outputs

# Process the OCR output to extract relevant medical information
def process_ocr_output(ocr_output):
    # Use a natural language processing library, such as spaCy, to extract
    # relevant medical information from the OCR output
    # (e.g., named entities, medical codes, patient demographics)
    nlp = spacy.load('en_core_web_sm')
    processed_output = []
    for text in ocr_output:
        doc = nlp(text)
        # Extract relevant information from the spaCy doc object
        processed_output.append(...)
    
    # Return the processed OCR output
    return processed_output

# Combine the OCR output with the output from the language model
def combine_ocr_and_language_outputs(ocr_output, language_output):
    # Combine the OCR output and the language model output to create a structured medical dataset
    medical_data = {}
    for i, text in enumerate(ocr_output):
        medical_data[i] = {'text': text, 'structured_info': language_output[i]}
    
    # Return the structured medical dataset
    return medical_data

# Convert the final output to JSON
def convert_to_json(medical_data):
    # Convert the structured medical dataset to a JSON string
    json_string = json.dumps(medical_data)
    
    # Write the JSON string to a file
    with open('medical_data.json', 'w') as f:
        f.write(json_string)


        
--====================----============---====================---==========---======---=====-----===========-----=======        
        
        
import cv2
import pytesseract
import re
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# OCR model for image enhancement and text extraction
def ocr(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 5)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return pytesseract.image_to_string(img)

# NLP model for text structuring and regeneration
def generate_medical_data(text):
    # Preprocessing the text data
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Structuring the text data using a pre-trained NLP model (BERT)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    structured_text = generator(text, max_length=100, do_sample=False)[0]['generated_text']
    
    # Regenerating medical data using the structured text
    data = []
    for i in range(10):
        generated_text = generator(structured_text, max_length=50, do_sample=True, temperature=0.5)[0]['generated_text']
        data.append(generated_text)
    
    # Converting the generated data to JSON for further analysis
    df = pd.DataFrame({'generated_data': data})
    json_data = df.to_json(orient='records')
    return json_data

# Testing the model
image_path = 'medical_note.jpg'
text = ocr(image_path)
generated_data = generate_medical_data(text)
print(generated_data)


'''
To build an alternative model to AWS Comprehend Medical, we need to perform natural language processing (NLP) tasks such as named entity recognition 
(NER), relationship extraction, and sentiment analysis on medical texts. Here is an example of Python code that can be used to build such a model:
'''

import spacy
from spacy import displacy
from spacy.tokens import Span
import pandas as pd
import numpy as np

# Load the pre-trained Spacy model
nlp = spacy.load('en_core_web_sm')

# Define medical entity labels
medical_entities = ['SYMPTOM', 'DISEASE', 'DRUG', 'GENE', 'CHEMICAL', 'PROCEDURE']

# Define custom entity matcher
def add_medical_entities(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    entity = Span(doc, start, end, label=match_id)
    doc.ents += (entity,)

# Add custom entity matcher to the pipeline
matcher = spacy.matcher.Matcher(nlp.vocab)
for entity in medical_entities:
    matcher.add(entity, add_medical_entities, [{'LOWER': entity.lower()}])

# Define a function to extract medical entities
def extract_medical_entities(text):
    doc = nlp(text)
    matches = matcher(doc)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Test the function
text = "The patient has been diagnosed with lung cancer and prescribed chemotherapy."
entities = extract_medical_entities(text)
print(entities)
