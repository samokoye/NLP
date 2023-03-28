'''
Given this medical notes below, how would you use it to make up to 1000 unique medical notes with a reasonable variation? Some of variations 
can be missing Doctor’s name but still make sense. Give me a Python code that can do it. 

"Following a thorough evaluation, Mr. Johnson was admitted to our medical facility on March 25th, 2023, with a diagnosis of pneumonia and 
hypertension. After administering appropriate treatment, we recommended a brief period of observation to monitor his progress."
'''
import random
import string

def generate_medical_notes(note, n):
    notes = []
    for i in range(n):
        # Generate variations in doctor's name
        doctors = ['Dr. Smith', 'Dr. Johnson', 'Dr. Brown', 'Dr. Davis', 'Dr. Wilson']
        doctor = random.choice(doctors)
        # Generate variations in illness
        illnesses = ['pneumonia', 'hypertension', 'diabetes', 'asthma', 'bronchitis']
        illness = random.choice(illnesses)
        # Generate variations in date
        year = random.randint(2021, 2023)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        date = f"{month}/{day}/{year}"
        # Generate variations in duration of hospital stay
        stay = random.choice(['a few days', 'a week', 'several weeks', 'a month'])
        # Generate a random string to add variation
        random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        # Replace the variables in the note with the generated variations
        new_note = note.replace('Mr. Johnson', f"Mr. {random_str}").replace('Dr. Smith', doctor).replace('pneumonia and hypertension', illness).replace('March 25th, 2023', date).replace('a brief period of observation', f"{stay} of observation")
        notes.append(new_note)
    return notes

# Example usage:
note = "Following a thorough evaluation, Mr. Johnson was admitted to our medical facility on March 25th, 2023, with a diagnosis of pneumonia and hypertension. After administering appropriate treatment, we recommended a brief period of observation to monitor his progress."
medical_notes = generate_medical_notes(note, 1000)
print(medical_notes[:5])  # Print the first 5 generated medical notes

############################
'''
update the code to use some kind of NLP or hugging face to show how you were able to recognize the NER in the given sentence 
'''
import random
import json
import pandas as pd
from transformers import pipeline

nlp = pipeline("ner", grouped_entities=True)

doctors = ["Dr. Smith", "Dr. Johnson", "Dr. Williams", "Dr. Brown", "Dr. Garcia"]
illnesses = ["pneumonia", "hypertension", "diabetes", "asthma", "cancer"]
facilities = ["Medical Center", "Hospital", "Clinic", "Center for Health"]
observations = ["24 hours", "48 hours", "72 hours", "5 days", "7 days"]

def generate_medical_note():
    doctor = random.choice(doctors)
    illness1, illness2 = random.sample(illnesses, 2)
    facility = random.choice(facilities)
    observation = random.choice(observations)
    
    note = f"Following a comprehensive evaluation, Mr. {random.choice(['Johnson', 'Smith', 'Williams', 'Brown', 'Garcia'])} was admitted to our {facility} on {random.randint(1, 28)}th of {random.choice(['March', 'April', 'May', 'June', 'July'])} 2023, with a diagnosis of {illness1} and {illness2}. After receiving appropriate treatment from {doctor}, we recommended a brief period of observation, which lasted for {observation}, to monitor his progress."
    
    ner_result = nlp(note)
    entities = {}
    for entity in ner_result:
        if entity["entity_group"] not in entities:
            entities[entity["entity_group"]] = []
        entities[entity["entity_group"]].append(entity["word"])
    
    return {"note": note, "entities": entities}

notes = [generate_medical_note() for _ in range(1000)]

# Write the notes to a JSON file
with open("medical_notes.json", "w") as f:
    json.dump(notes, f)

# Convert the notes to a Pandas DataFrame
df = pd.json_normalize(notes, "entities", ["note"])
df = df.pivot_table(index="note", columns="entity_group", values="0", aggfunc=" ".join)
df.reset_index(inplace=True)
df.to_csv("medical_notes.csv", index=False)
#################################
'''
here's an updated code that will randomly omit the doctor's name in some of the sentences while still maintaining the meaning and 
uniqueness of the sentence structure:

'''
import random
import json
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

# List of illness and doctor names
illnesses = ["pneumonia", "hypertension", "diabetes", "arthritis", "migraine", "depression", "asthma", "cancer"]
doctors = ["Dr. Smith", "Dr. Johnson", "Dr. Brown", "Dr. Lee", "Dr. Davis", "Dr. Wilson", "Dr. Baker", "Dr. Carter"]

# List of sentence structures
structures = [
    "After being diagnosed with {illness} on {date}, Mr./Mrs. {surname} was admitted to our medical facility for treatment.",
    "We admitted Mr./Mrs. {surname} to our hospital on {date} due to a diagnosis of {illness}.",
    "Following a thorough evaluation, Mr./Mrs. {surname} was diagnosed with {illness} and admitted to our medical facility on {date}.",
    "Upon arriving at our medical facility on {date}, Mr./Mrs. {surname} was diagnosed with {illness} and admitted for treatment.",
    "Mr./Mrs. {surname} was admitted to our hospital on {date} after being diagnosed with {illness}."
]

# Generate 1000 unique medical notes
medical_notes = []
for i in range(1000):
    illness = random.choice(illnesses)
    doctor = random.choice(doctors)
    structure = random.choice(structures)
    date = f"{random.randint(1, 28)}-{random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])}-2023"
    surname = nlp(illness).ents[0].text.capitalize() + str(random.randint(1, 1000))
    
    # Replace doctor's name in sentence structure if it exists
    if "{doctor}" in structure:
        structure = structure.replace("{doctor}", doctor)
    else:
        doctor = None
    
    # Format sentence
    sentence = structure.format(illness=illness, date=date, surname=surname)
    
    # Add optional doctor's name to medical note
    if doctor:
        medical_note = {"sentence": sentence, "illness": illness, "doctor": doctor, "date": date, "surname": surname}
    else:
        medical_note = {"sentence": sentence, "illness": illness, "date": date, "surname": surname}
        
    medical_notes.append(medical_note)
    
# Save medical notes as JSON file
with open("medical_notes.json", "w") as f:
    json.dump(medical_notes, f)

# Convert medical notes to DataFrame (optional)
df = pd.DataFrame(medical_notes)
########################################

'''
10 unique sentence structures
'''
# Given medical note
med_note = "On May 2nd, 2023, Mr. Patel was transferred to our facility with a diagnosis of acute myocardial infarction. We immediately began administering treatment and monitoring his condition. After several days of observation, we recommended a course of physical therapy to aid in his recovery."

# Extracting named entities from the given medical note
doc = nlp(med_note)
entities = [(ent.text, ent.label_) for ent in doc.ents]

# Unique sentence structures
structures = [
    "On {date}, {name} was admitted to our medical facility with a diagnosis of {illness}. {doctor} immediately administered appropriate treatment and began monitoring {gender} condition. After several days of observation, a course of {therapy} was recommended to aid in {gender} recovery.",
    "{name} was transferred to our medical center on {date} with a diagnosis of {illness}. {doctor} began treatment and closely monitored {gender} progress. After a few days of observation, {therapy} was prescribed to aid in {gender} recovery.",
    "We received {name} at our hospital on {date} with a diagnosis of {illness}. {doctor} started the necessary treatment and observed {gender} progress for a few days. We then recommended {therapy} to aid in {gender} recovery.",
    "{name} was admitted to our medical facility on {date} with a diagnosis of {illness}. {doctor} administered appropriate treatment and monitored {gender} progress closely. Following observation for several days, we prescribed {therapy} to aid in {gender} recovery.",
    "Following a thorough evaluation, {name} was admitted to our hospital on {date} with a diagnosis of {illness}. {doctor} started the necessary treatment and closely monitored {gender} progress. After a few days of observation, we prescribed {therapy} to aid in {gender} recovery.",
    "{name} was transferred to our hospital on {date} with a diagnosis of {illness}. {doctor} started the necessary treatment and monitored {gender} condition closely. After observation for several days, we recommended {therapy} to aid in {gender} recovery.",
    "We received {name} at our medical center on {date} with a diagnosis of {illness}. {doctor} administered appropriate treatment and closely monitored {gender} progress. After several days of observation, we prescribed {therapy} to aid in {gender} recovery.",
    "{name} was admitted to our medical center on {date} with a diagnosis of {illness}. {doctor} began the necessary treatment and observed {gender} condition closely. Following a few days of observation, {therapy} was prescribed to aid in {gender} recovery.",
    "On {date}, {name} was transferred to our hospital with a diagnosis of {illness}. {doctor} started the necessary treatment and closely monitored {gender} progress. Following several days of observation, we recommended {therapy} to aid in {gender} recovery.",
    "We admitted {name} to our medical center on {date} with a diagnosis of {illness}. {doctor} began the necessary treatment and monitored {gender} condition closely. After several days of observation, we prescribed {therapy} to aid in {gender} recovery."
]


#DRAFT
illnesses = ["pneumonia", "hypertension", "acute myocardial infarction", "asthma", "diabetes", "malaria", "influenza", "cancer", "arthritis", "bronchitis"]
treatments = ["antibiotics", "pain relievers", "steroids", "antidepressants", "blood pressure medication", "insulin", "chemotherapy", "immunotherapy", "radiation therapy", "surgery"]
observation_durations = ["a few days", "several days", "a week", "two weeks", "three weeks", "a month", "several months", "a year", "several years"]
physical_therapy_durations = ["a few days", "a week", "two weeks", "three weeks", "a month", "several months", "a year", "several years"]
admitted_sentences = [
    "Admitted to our facility on {admission_date}, Mr./Ms. {last_name} presented with a diagnosis of {illness}.",
    "On {admission_date}, we received Mr./Ms. {last_name} into our care, who was diagnosed with {illness}.",
    "Mr./Ms. {last_name} was admitted on {admission_date} to our facility with a diagnosis of {illness}.",
    "We received Mr./Ms. {last_name} at our facility on {admission_date} and diagnosed them with {illness}.",
    "Following an evaluation on {admission_date}, Mr./Ms. {last_name} was admitted to our facility with a diagnosis of {illness}.",
    "Our team admitted Mr./Ms. {last_name} to our facility on {admission_date} after diagnosing them with {illness}.",
    "Mr./Ms. {last_name} was brought to our facility on {admission_date} with a diagnosis of {illness} and admitted for further care.",
    "On {admission_date}, we admitted Mr./Ms. {last_name} to our facility with a diagnosis of {illness}.",
    "We received Mr./Ms. {last_name} at our facility on {admission_date} for treatment of {illness}.",
    "Admitted to our facility on {admission_date}, Mr./Ms. {last_name} was diagnosed with {illness} and received immediate care from our team."
]

observation_sentences = [
    "After receiving treatment, we recommended a period of observation lasting {duration} to monitor Mr./Ms. {last_name}'s progress.",
    "We closely monitored Mr./Ms. {last_name}'s condition over the course of {duration} after administering treatment.",
    "Following initial treatment, we advised that Mr./Ms. {last_name} stay with us for {duration} to ensure they were recovering as expected.",
    "To keep a close eye on Mr./Ms. {last_name}'s recovery, we recommended they remain under observation at our facility for {duration} after treatment.",
    "After administering appropriate care, we recommended a period of observation lasting {duration} to ensure Mr./Ms. {last_name}'s recovery.",
    "Our team recommended that Mr./Ms. {last_name} stay under observation at our facility for {duration} after receiving treatment.",
    "To ensure the best possible outcome, we advised Mr./Ms. {last_name} to remain..]
  





