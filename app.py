from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from datetime import datetime
from pydantic import BaseModel
import os
from huggingface_hub import login
import re
from sklearn.metrics.pairwise import cosine_similarity


# ✅ Initialize FastAPI
app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)
# ✅ Load Model
model_name = "sche0196/Sres_2"  # Replace with your model

# Load quantization config (only needed for 4-bit or 8-bit models)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=False,  # Set to True if using 8-bit quantization
    load_in_4bit=True,  # Set to True if using 4-bit quantization
    llm_int8_threshold=6.0
)

# Load the model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()  # Set to evaluation mode

# ✅ Load FAISS Index & Symptom-to-Disease Data
INDEX_PATH = "/New_symptom_disease_index.faiss"
df = pd.read_csv("New_Combined_Symptom2Disease.csv")
index = faiss.read_index(INDEX_PATH)

# ✅ Load Sentence Transformer
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
MONGO_URI = os.getenv("MONGO_URI")
# ✅ Connect to MongoDB
mongo_client = MongoClient(MONGO_URI)
chat_db = mongo_client["telemedicine"]
chat_collection = chat_db["chat_history_ft"]

# ✅ Pydantic Model
class DiagnosisRequest(BaseModel):
    patient_id: str
    user_input: str

# ✅ Helper Functions
def extract_relevant_symptoms(patient_id):
    """Extract relevant symptom inputs from chat history."""
    chat = chat_collection.find_one({"patient_id": patient_id})
    if not chat or "messages" not in chat:
        return ""
    return " ".join(
        msg["message"].lower()
        for msg in sorted(chat["messages"], key=lambda x: x["timestamp"])
        if msg["role"] == "user" and msg["message"].lower() not in ["hi", "hello", "hey"] and "thank" not in msg["message"].lower()
    )

def extract_most_likely_disease(response, retrieved_diseases):
    """Extract the most likely diagnosis from the model's response."""
    match = re.search(r"(?i)most likely diagnosis:\s*([\w\s-]+)", response)
    if match:
        return match.group(1).strip()
    for disease in retrieved_diseases:     # If no explicit mention, find the closest match from retrieved diseases
        if disease.lower() in response.lower():
            return disease  # Return the first match
    return None  # No clear diagnosis found

def is_general_disease_query(user_input):
    """Detects if the user is asking about a disease rather than presenting symptoms."""
    pattern = r"\b(?:what is|tell me about|explain|define|symptoms of|causes of|treatment for)\s+([\w\s-]+)\b"
    match = re.search(pattern, user_input.lower())
    return match.group(1).strip() if match else None


def update_diagnosis_count(patient_id, disease):
    """Update MongoDB to track how many times the same disease was diagnosed."""
    if not disease:
        return  # Skip if no clear disease found
    normalized_disease = disease.strip().lower()
    # Increment count for this disease
    chat_collection.update_one(
        {"patient_id": patient_id},
        {"$inc": {f"diagnosis_counts.{normalized_disease}": 1}},
        upsert=True  # Create entry if it doesn’t exist
    )

def get_diagnosis_count(patient_id, disease):
    """Retrieve the count of a specific diagnosis for this patient."""
    normalized_disease = disease.strip().lower()
    chat = chat_collection.find_one({"patient_id": patient_id}) or {}
    return chat.get("diagnosis_counts", {}).get(normalized_disease, 0)


def save_chat(patient_id, role, message):
    chat_collection.update_one(
        {"patient_id": patient_id},
        {"$push": {"messages": {"role": role, "message": message, "timestamp": datetime.utcnow()}}},
        upsert=True,
    )
    
def is_repeating_diagnosis(assistant_response, past_responses):
    """Check if the assistant is repeating the same reasoning using semantic similarity."""
    if len(past_responses) < 3:  # Only compare if we have enough history
        return False  

    # Convert responses into embeddings
    response_embeddings = embed_model.encode([assistant_response] + past_responses[-2:], normalize_embeddings=True)

    # Compute cosine similarities
    similarities = cosine_similarity([response_embeddings[0]], response_embeddings[1:])[0]

    # If both previous responses have high similarity (>0.85), it's repetitive
    return all(sim > 0.85 for sim in similarities)


def retrieve_diseases(user_input, top_k=5):    
    input_embedding = embed_model.encode([user_input], convert_to_numpy=True, normalize_embeddings=True)
    _, indices = index.search(input_embedding, top_k * 2)  
    retrieved = []
    seen = set()
    for i in indices[0]:
        if i < len(df):
            label = df.iloc[i]["label"]
            if label not in seen:
                seen.add(label)
                retrieved.append({
                    "symptoms": df.iloc[i]["text"],
                    "disease": label
                })
            if len(retrieved) >= top_k:
                break

    return retrieved

def format_prompt(user_input, retrieved_diseases):
    """Format prompt with past reasoning to prevent repetition."""
    disease_info = "\n".join([f"- {d}" for d in retrieved_diseases])

    return f"""You are a highly knowledgeable medical assistant specializing in diagnosing diseases.

    User Symptoms:
    "{user_input}"

    Based on a similarity search, here are related examples of symptoms and their associated diseases:
    {disease_info}

    Please:
    1. Breifly mention a few possible diagnoses but state the most likely diagnosis.
    2. Ask a follow-up question based on what the user has told to refine the diagnosis without making assumptions.
    3. Do NOT include excessive explanations, additional follow-ups, step-by-step breakdowns, or random text.
    4. The response should be concise and one to two sentences long.
    5. If the {user_input} is a general question, do not respond with a diagnosis but resopnd with a general answer.

    Assistant:"""
    
def get_specialist_recommendation(disease):
    """Map diseases to appropriate medical specialists."""
    sp_df = pd.read_csv("Doctor_Versus_Disease.csv", encoding='windows-1254')
    sp_df['Drug Reaction'] = sp_df['Drug Reaction'].str.lower()
    disease = disease.lower()
    # Directly filter without extracting index
    result = sp_df.loc[sp_df['Drug Reaction'] == disease, 'Allergist']
    
    return result.iloc[0] if not result.empty else "General Physician for Further Clarity" 

def get_diagnosis(patient_id, user_input):
    """Handles diagnosis, response generation, and stops after repeated diagnoses."""

    if user_input.lower() in ["hi", "hello"]:
        return "Hello! How can I assist you today?"
    if "thank" in user_input.lower():
        return "You're welcome! Let me know if you have more questions."

    full_input = extract_relevant_symptoms(patient_id) + " " + user_input
    retrieved_diseases = retrieve_diseases(full_input)

    if not retrieved_diseases:
        return "I couldn't find relevant diseases. Can you describe your symptoms further?"

    prompt = format_prompt(full_input, retrieved_diseases)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        repetition_penalty=1.2,
        temperature=0.7,
        top_p=0.85,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()

    most_likely_disease = extract_most_likely_disease(response, [d['disease'] for d in retrieved_diseases])

    if most_likely_disease:
        update_diagnosis_count(patient_id, most_likely_disease)
        # **Step 3: Stop Follow-Ups if the Same Disease is Repeated**
        diagnosis_count = get_diagnosis_count(patient_id, most_likely_disease)
        if diagnosis_count >= 3:
            specialist = get_specialist_recommendation(most_likely_disease)
            response = (f"I've identified your condition as **{most_likely_disease}** based on your symptoms. "
                        f"I strongly recommend booking an appointment with a **{specialist}** for further evaluation.")

    save_chat(patient_id, "user", user_input)
    save_chat(patient_id, "assistant", response)

    return response




@app.post("/predict")
async def diagnose(request: DiagnosisRequest):
    return {"response": get_diagnosis(request.patient_id, request.user_input)}
