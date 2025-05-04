from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import easyocr
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import os
from openai import OpenAI
import json
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration OpenAI
client = OpenAI(api_key=os.getenv("OPEN_AI"))

# Configuration EasyOCR
reader = easyocr.Reader(['fr', 'en'])

app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DocumentAnalysis(BaseModel):
    document_type: str
    confidence: float
    details: dict

class AccountingEntry(BaseModel):
    date: str
    description: str
    amount: float
    account_type: str
    confidence: float

def extract_text_from_image(image):
    """Extraire le texte d'une image en utilisant EasyOCR et Tesseract"""
    # EasyOCR pour la détection principale
    easy_result = reader.readtext(image)
    easy_text = " ".join([text[1] for text in easy_result])
    
    # Tesseract comme backup pour la vérification
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    tesseract_text = pytesseract.image_to_string(image, lang='fra+eng')
    
    # Combiner les résultats
    combined_text = f"{easy_text}\n{tesseract_text}"
    return combined_text

def analyze_document_type(text: str) -> DocumentAnalysis:
    """Analyser le type de document comptable avec OpenAI"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
               {
                    "role": "system",
                    "content": """You are an expert in analyzing accounting documents. Analyze the provided text and categorize the document into one of the following types:
                    - CLIENTS
                    - FOURNISSEURS
                    - NOTE DE FRAIS
                    - CAISSE
                    - BANQUES
                    - SOCIAL
                    - FISCAL
                    - COURRIERS
                    - ILLISIBLES
                    - GESTION
                    - ETATS COMPTABLES

                    Respond in JSON using the following structure:
                    {
                        "document_type": "type_of_document",
                        "confidence": 0.XX,
                        "details": {
                            "issuer": "name_if_available",
                            "date": "date_if_available",
                            "total_amount": "amount_if_available"
                        }
                    }

                    Ensure the returned JSON is properly formatted and valid. If any information is unavailable, use `null` as the value. and don't add ```json just return the JSON stringfy
                    """
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )
        result = json.loads(response.choices[0].message.content)
        return DocumentAnalysis(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse du type de document: {str(e)}")

def extract_accounting_entries(text: str) -> List[AccountingEntry]:
    """Extraire les écritures comptables avec OpenAI"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
               {
                    "role": "system",
                    "content": """You are an accounting expert. Analyze the provided text and extract all accounting entries.
                    For each entry, identify:
                    - The date
                    - The description
                    - The amount
                    - The account type (debit/credit)

                    Respond in JSON with an array of entries using the following structure:
                    [
                        {
                            "date": "date_of_entry",
                            "description": "description_of_entry",
                            "confidence": 0.XX,
                            "amount": "amount_of_entry",
                            "account_type": "debit_or_credit"
                        }
                    ]

                    Ensure the returned JSON is properly formatted and valid. If any information is unavailable, use `null` as the value and don't add ```json just return the JSON stringfy,
                    don't add anything else
                    """
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )
        print(response.choices[0].message.content)
        result = json.loads(response.choices[0].message.content)
        return [AccountingEntry(**entry) for entry in result]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'extraction des écritures: {str(e)}")

@app.post("/categorize-document")
async def categorize_document(file: UploadFile) -> DocumentAnalysis:
    """Endpoint pour catégoriser un document comptable"""
    try:
        content = await file.read()
        
        # Si c'est un PDF, convertir en images
        if file.filename.lower().endswith('.pdf'):
            images = convert_from_bytes(content)
            text = ""
            for image in images:
                # Convertir l'image PIL en bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                text += extract_text_from_image(img_byte_arr) + "\n"
        else:
            text = extract_text_from_image(content)
        
        return analyze_document_type(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-entries")
async def extract_entries(file: UploadFile) -> List[AccountingEntry]:
    """Endpoint pour extraire les écritures comptables d'un document"""
    try:
        content = await file.read()
        
        # Si c'est un PDF, convertir en images
        if file.filename.lower().endswith('.pdf'):
            images = convert_from_bytes(content)
            text = ""
            for image in images:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                text += extract_text_from_image(img_byte_arr) + "\n"
        else:
            text = extract_text_from_image(content)
        
        return extract_accounting_entries(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
