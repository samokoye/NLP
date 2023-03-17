#1
'''
Extracting demographic information from a medical handwritten note. There are several NLP libraries and models available in Python that can help with 
this task. One such popular library is SpaCy, which offers pre-trained models for named entity recognition (NER).
'''
import spacy

# load SpaCy's pre-trained NER model
nlp = spacy.load("en_core_web_sm")

# define a function to extract demographic information
def extract_demographics(text):
    # create a Doc object
    doc = nlp(text)

    # define a dictionary to store the extracted information
    demographics = {}

    # iterate over the entities in the document
    for ent in doc.ents:
        # check if the entity is a demographic information
        if ent.label_ in ["PERSON", "DATE", "GPE"]:
            # add the entity to the dictionary
            if ent.label_ in demographics:
                demographics[ent.label_].append(ent.text)
            else:
                demographics[ent.label_] = [ent.text]

    # return the dictionary
    return demographics

# example usage
text = "Patient is a 32-year-old male from San Francisco, California."
demographics = extract_demographics(text)
print(demographics)


#2

import spacy

nlp = spacy.load("en_core_web_sm")

def extract_demographics(note):
    doc = nlp(note)
    demographics = {
        "Name": [],
        "Age": [],
        "Gender": [],
        "Address": [],
        "Phone": [],
        "Email": [],
        "Medical Record Number": [],
        "Insurance Information": []
    }
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            demographics["Name"].append(ent.text)
        elif ent.label_ == "AGE":
            demographics["Age"].append(ent.text)
        elif ent.label_ == "GENDER":
            demographics["Gender"].append(ent.text)
        elif ent.label_ == "ADDRESS":
            demographics["Address"].append(ent.text)
        elif ent.label_ == "PHONE":
            demographics["Phone"].append(ent.text)
        elif ent.label_ == "EMAIL":
            demographics["Email"].append(ent.text)
        elif ent.label_ == "MEDICAL_RECORD_NUMBER":
            demographics["Medical Record Number"].append(ent.text)
        elif ent.label_ == "INSURANCE_INFO":
            demographics["Insurance Information"].append(ent.text)
    return demographics

# Example usage
note = "Patient Name: John Doe\nAge: 45\nGender: Male\nAddress: 123 Main St\nPhone: 555-555-5555\nEmail: johndoe@email.com\nMedical Record Number: 12345\nInsurance Information: XYZ Health"
demographics = extract_demographics(note)
print(demographics)


'''
Assuming the medical file is a pdf
Here's an example Python code that uses the PyPDF2 library to extract the text from a PDF file, and then uses the Hugging Face Transformers 
library to perform topic modeling on the extracted text:
'''
import PyPDF2
from transformers import pipeline

# Load the PDF file
with open('example.pdf', 'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    # Extract the text from all pages of the PDF file
    text = ''
    for page in range(pdf_reader.getNumPages()):
        text += pdf_reader.getPage(page).extractText()

# Perform topic modeling on the extracted text
nlp = pipeline("text2text-generation", model="valhalla/t5-base-qa-summary")
result = nlp(text, max_length=100, do_sample=True, top_p=0.95, top_k=60)

# Categorize the texts into topics
topics = {}
for item in result[0]['generated_text'].split('\n'):
    if item.strip() != '':
        key = item.strip()
        topics[key] = ''
    else:
        topics[key] += item.strip()

# Print the categorized texts
print(topics)

'''
This code first loads the PDF file using PyPDF2 library and extracts the text from all pages. Then it uses the Hugging Face Transformers library to 
perform topic modeling on the extracted text. We use the text2text-generation pipeline and the valhalla/t5-base-qa-summary model to generate a summary 
of the text. The max_length, do_sample, top_p, and top_k parameters are set to tune the quality of the generated summary.

Finally, we categorize the generated summary into topics using a simple approach. We split the generated summary into lines, and if a line is not empty, 
we use it as a key in the topics dictionary. If a line is empty, we append the line to the current value of the most recently added key. This assumes 
that each key in the generated summary corresponds to a topic, and that the subsequent lines under each key belong to that topic.

The code then prints the topics dictionary, which contains the categorized texts, with the topics as keys and the texts as values. 
'''
