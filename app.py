from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import zipfile
import io
import logging
import tempfile
import time
from typing import List, Dict, Any, Optional, Set
import pandas as pd
import psycopg2
from psycopg2.extras import Json, DictCursor
from datetime import datetime
import uuid
import shutil
import json
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import re
import uvicorn
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Processing API",
    description="API for processing zip files containing different document types using OCR and Gemini",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DB_CONFIG = {
    'dbname': 'batch_document_processing',
    'user': 'soubhikghosh',  # Replace with your username
    'password': '99Ghosh',  # Replace with your password
    'host': 'localhost',
    'port': '5432'
}

# Configure Google API - replace with your API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCD6DGeERwWQbBC6BK1Hq0ecagQj72rqyQ")
genai.configure(api_key=GOOGLE_API_KEY)

# Document types
DOCUMENT_TYPES = [
    "customer_request_letter",
    "form_15ca",
    "form_15cb",
    "form_a2",
    "invoice",
    "transport_document", 
    "fdd_stationery"
]

# Field definitions with their sources
FIELDS = [
    {"id": 1, "name": "currency", "source": "customer_request_letter"},
    {"id": 2, "name": "amount", "source": "customer_request_letter"},
    {"id": 3, "name": "beneficiary_account_number", "source": "customer_request_letter"},
    {"id": 4, "name": "beneficiary_name", "source": "customer_request_letter"},
    {"id": 5, "name": "beneficiary_address", "source": "customer_request_letter"},
    {"id": 6, "name": "beneficiary_bank_swift_code", "source": "customer_request_letter"},
    {"id": 7, "name": "beneficiary_bank_name", "source": "customer_request_letter"},
    {"id": 8, "name": "beneficiary_bank_address", "source": "customer_request_letter"},
    {"id": 9, "name": "charge_type", "source": "customer_request_letter"},
    {"id": 10, "name": "dd_number", "source": "fdd_stationery"},
    {"id": 11, "name": "intermediary_institution", "source": "customer_request_letter"},
    {"id": 12, "name": "ack_no_form_15ca", "source": "form_15ca"},
    {"id": 13, "name": "ack_no_form_15cb", "source": "form_15cb"},
    {"id": 14, "name": "account_to_be_debited_for_remittance", "source": "customer_request_letter"},
    {"id": 15, "name": "account_to_be_debited_for_charges", "source": "customer_request_letter"},
    {"id": 16, "name": "remittance_account", "source": "customer_request_letter"},
    {"id": 17, "name": "invoice_number", "source": "invoice"},
    {"id": 18, "name": "invoice_date", "source": "invoice"},
    {"id": 19, "name": "transport_document_number", "source": "transport_document"},
    {"id": 20, "name": "transport_document_date", "source": "transport_document"},
    {"id": 21, "name": "port_of_loading", "source": "transport_document"},
    {"id": 22, "name": "port_of_discharge", "source": "transport_document"},
    {"id": 23, "name": "on_board_date", "source": "transport_document"},
    {"id": 24, "name": "remittance_information", "source": "customer_request_letter"},
    {"id": 25, "name": "purpose_code", "source": "form_a2"},
    {"id": 26, "name": "form_15ca", "source": "form_15ca"},
    {"id": 27, "name": "customer_request_letter_date", "source": "customer_request_letter"},
    {"id": 28, "name": "payment_reference_drawer", "source": "customer_request_letter"},
    {"id": 29, "name": "goods_description", "source": "invoice"},
    {"id": 30, "name": "inco_terms", "source": "invoice"},
    {"id": 31, "name": "goods_carrier", "source": "transport_document"},
    {"id": 32, "name": "goods_shipment_date", "source": "transport_document"},
    {"id": 33, "name": "bullion", "source": "invoice"},
    {"id": 34, "name": "bullion_customer", "source": "invoice"},
    {"id": 35, "name": "bullion_delivery", "source": "invoice"},
    {"id": 36, "name": "bullion_weight", "source": "invoice"}
]

# Create a mapping of document types to their fields
DOCUMENT_FIELDS = {}
for field in FIELDS:
    doc_type = field["source"]
    if doc_type not in DOCUMENT_FIELDS:
        DOCUMENT_FIELDS[doc_type] = []
    DOCUMENT_FIELDS[doc_type].append(field["name"])

class DocumentProcessor:
    """Helper class for document processing operations"""
    
    @staticmethod
    def perform_ocr(file_data: bytes, file_type: str) -> Dict[str, Any]:
        """Perform OCR on the file data based on file type."""
        try:
            result = {"text": "", "pages": []}
            
            if file_type.lower() in ["image/jpeg", "image/jpg", "image/png", "image/tiff"]:
                # Process image directly
                image = Image.open(io.BytesIO(file_data))
                text = pytesseract.image_to_string(image)
                result["text"] = text
                result["pages"].append({"page_num": 1, "text": text})
                
            elif file_type.lower() in ["application/pdf", "pdf"]:
                # Convert PDF to images and process each page
                images = convert_from_bytes(file_data)
                full_text = ""
                
                for i, image in enumerate(images):
                    page_text = pytesseract.image_to_string(image)
                    full_text += page_text + "\n\n"
                    result["pages"].append({"page_num": i+1, "text": page_text})
                
                result["text"] = full_text
                
            else:
                logger.warning(f"Unsupported file type for OCR: {file_type}")
                result["text"] = "Unsupported file type for OCR"
                result["error"] = f"Unsupported file type: {file_type}"
                
            return result
            
        except Exception as e:
            logger.error(f"Error during OCR processing: {str(e)}")
            return {"error": str(e), "text": "", "pages": []}
    
    @staticmethod
    def classify_document(text: str) -> Dict[str, Any]:
        """Classify the document type using Gemini AI."""
        try:
            model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
            
            prompt = """
            You are a document classification expert. Examine the following text extracted via OCR and determine which type of financial document it is.
            
            Classify the document into EXACTLY ONE of these categories:
            - customer_request_letter
            - form_15ca
            - form_15cb
            - form_a2
            - invoice
            - transport_document
            - fdd_stationery
            - unknown
            
            Look for specific headers, formatting patterns, and content that uniquely identify each document type:
            
            ### Customer Request Letter
            - Typically contains: "Request for remittance", "Outward remittance", or similar phrases
            - Usually has beneficiary details, bank details, and remittance amount
            - Often contains a formal request structure with date and signature
            
            ### Form 15CA
            - Official Indian tax form for foreign remittances
            - Contains "FORM 15CA" in the header
            - Has sections related to Income Tax Act and remittance declarations
            - Contains an acknowledgment number
            
            ### Form 15CB
            - Certificate from Chartered Accountant related to foreign remittances
            - Contains "FORM 15CB" in the header
            - Has CA certification and registration numbers
            
            ### Form A2
            - Foreign exchange transaction form
            - Contains "FORM A2" in the header
            - Has sections for purpose codes and forex transaction details
            
            ### Invoice
            - Contains line items with quantities, unit prices, and totals
            - Has invoice number, date, and payment terms
            - Lists buyer and seller information
            
            ### Transport Document
            - Bill of Lading (B/L), Airway Bill, or similar
            - Contains shipping details, ports, vessel information
            - Has consignor and consignee information
            
            ### FDD Stationery
            - Foreign Demand Draft document
            - Contains DD number and banking instrument details
            
            Provide your classification as a simple JSON with two fields:
            {
                "document_type": "one of the categories listed above",
                "confidence": 0.XX (a number between 0 and 1)
            }
            
            Return ONLY this JSON with no explanations or additional text.
            """
            
            classification_response = model.generate_content([prompt, text])
            json_str = classification_response.text.strip()
            
            # Clean up potential JSON formatting
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
                
            classification = json.loads(json_str.strip())
            return classification
            
        except Exception as e:
            logger.error(f"Error during document classification: {str(e)}")
            return {"document_type": "unknown", "confidence": 0.0, "error": str(e)}
    
    @staticmethod
    def extract_fields(text: str, document_type: str) -> Dict[str, Any]:
        """Extract fields from document using Gemini AI."""
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Get fields for this document type
            doc_fields = DOCUMENT_FIELDS.get(document_type, [])
            if not doc_fields:
                return {
                    "extracted_fields": [],
                    "metadata": {
                        "analysis_timestamp": datetime.now().isoformat(),
                        "overall_confidence": 0.0,
                        "error": f"No fields defined for document type: {document_type}"
                    }
                }
            
            # Define field descriptions for better extraction
            field_descriptions = {
                "currency": "The currency for the transaction (e.g., USD, EUR, GBP)",
                "amount": "The amount of money being transferred",
                "beneficiary_account_number": "The bank account number of the recipient",
                "beneficiary_name": "The name of the person or entity receiving the funds",
                "beneficiary_address": "The address of the beneficiary",
                "beneficiary_bank_swift_code": "The SWIFT/BIC code of the beneficiary's bank",
                "beneficiary_bank_name": "The name of the beneficiary's bank",
                "beneficiary_bank_address": "The address of the beneficiary's bank",
                "charge_type": "The type of charges for the transaction (OUR, BEN, SHA)",
                "dd_number": "The Demand Draft number printed on the FDD stationery",
                "intermediary_institution": "Any intermediary bank involved in the transaction",
                "ack_no_form_15ca": "The acknowledgment number on Form 15CA",
                "ack_no_form_15cb": "The acknowledgment number on Form 15CB",
                "account_to_be_debited_for_remittance": "The account from which the main amount will be taken",
                "account_to_be_debited_for_charges": "The account from which the fees will be taken",
                "remittance_account": "The account for the remittance",
                "invoice_number": "The identification number of the invoice",
                "invoice_date": "The date when the invoice was issued",
                "transport_document_number": "The number on the transport document",
                "transport_document_date": "The date on the transport document",
                "port_of_loading": "The port where goods were loaded",
                "port_of_discharge": "The port where goods will be unloaded",
                "on_board_date": "The date when goods were loaded on vessel",
                "remittance_information": "Details about the purpose of the remittance",
                "purpose_code": "The code indicating the purpose of the foreign exchange transaction",
                "form_15ca": "Details from the Form 15CA document",
                "customer_request_letter_date": "The date on the customer request letter",
                "payment_reference_drawer": "Reference information for the payment drawer",
                "goods_description": "Description of the goods on the invoice",
                "inco_terms": "International Commercial Terms on the invoice (e.g., FOB, CIF)",
                "goods_carrier": "The carrier/vessel transporting the goods",
                "goods_shipment_date": "The date when goods were shipped",
                "bullion": "Whether the invoice is for bullion (yes/no)",
                "bullion_customer": "The customer for bullion transaction",
                "bullion_delivery": "Delivery details for bullion",
                "bullion_weight": "Weight of bullion being transacted"
            }
            
            # Create field list with descriptions
            fields_with_descriptions = []
            for field in doc_fields:
                description = field_descriptions.get(field, "")
                fields_with_descriptions.append(f"- {field}: {description}")
            
            fields_list = "\n".join(fields_with_descriptions)
            
            prompt = f"""
            You are a specialized financial document analyzer with expertise in extracting information from {document_type.replace("_", " ").title()} documents.
            
            Extract the following fields from the document with maximum precision:
            {fields_list}
            
            For each field:
            1. Extract the exact value as it appears in the document
            2. If the text is unclear due to OCR issues, make a reasonable approximation
            3. For dates, standardize to YYYY-MM-DD format where possible
            4. For monetary amounts, include both value and currency
            5. Assign a confidence score between 0.0 and 1.0 for each extraction
            6. If a field cannot be found or extracted, provide a reason why
            
            For each field, provide:
            - The extracted value
            - A confidence score (0.0-1.0)
            - The exact text segment from which you extracted the information
            - A reason if the field couldn't be extracted
            
            Provide your analysis in the following strict JSON structure only:
            {{
                "extracted_fields": [
                    {{
                        "field_name": "field_name",
                        "value": "extracted value",
                        "confidence": 0.XX,
                        "source_text": "text from document",
                        "reason": "reason if field couldn't be extracted properly"
                    }}
                ],
                "metadata": {{
                    "analysis_timestamp": "ISO timestamp",
                    "overall_confidence": 0.XX
                }}
            }}
            
            Return ONLY valid JSON with no explanations or additional text.
            """
            
            extraction_response = model.generate_content([prompt, text])
            json_str = extraction_response.text.strip()
            
            # Clean up potential JSON formatting
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
                
            extraction_result = json.loads(json_str.strip())
            return extraction_result
            
        except Exception as e:
            logger.error(f"Error during field extraction: {str(e)}")
            return {
                "extracted_fields": [],
                "metadata": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "overall_confidence": 0.0,
                    "error": str(e)
                }
            }

def get_db_connection():
    """Create the database if it doesn't exist and return a connection."""
    try:
        # First try to connect to the default postgres database to check if our database exists
        conn = psycopg2.connect(
            dbname='postgres',
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port']
        )
        conn.autocommit = True
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Check if our database exists
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_CONFIG['dbname'],))
        exists = cursor.fetchone()
        
        # Create database if it doesn't exist
        if not exists:
            logger.info(f"Database '{DB_CONFIG['dbname']}' does not exist. Creating...")
            cursor.execute(f"CREATE DATABASE {DB_CONFIG['dbname']}")
            logger.info(f"Database '{DB_CONFIG['dbname']}' created successfully")
        
        cursor.close()
        conn.close()
        
        # Now connect to our actual database
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info(f"Connected to database '{DB_CONFIG['dbname']}'")
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

def init_db():
    """Initialize database tables if they don't exist."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Create processed_files table to track uploaded files
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_files (
            id SERIAL PRIMARY KEY,
            job_id TEXT,
            file_name TEXT,
            folder_name TEXT,
            file_path TEXT,
            document_type TEXT,
            confidence FLOAT,
            processing_status TEXT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
        ''')
        
        # Create extracted_fields table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS extracted_fields (
            id SERIAL PRIMARY KEY,
            job_id TEXT,
            file_id INTEGER REFERENCES processed_files(id),
            field_name TEXT,
            field_value TEXT,
            confidence FLOAT,
            reason TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
        ''')
        
        # Create processing_jobs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS processing_jobs (
            id SERIAL PRIMARY KEY,
            job_id TEXT UNIQUE,
            status TEXT,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            total_files INTEGER,
            processed_files INTEGER,
            output_file_path TEXT,
            error_message TEXT
        )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_processed_files_job_id ON processed_files(job_id);
        CREATE INDEX IF NOT EXISTS idx_extracted_fields_job_id ON extracted_fields(job_id);
        CREATE INDEX IF NOT EXISTS idx_extracted_fields_file_id ON extracted_fields(file_id);
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise

async def process_zip_files(file_contents: List[bytes], file_names: List[str], job_id: str):
    """Process multiple zip files and generate Excel report."""
    try:
        # Create a temp directory for this job
        temp_dir = tempfile.mkdtemp(prefix=f"job_{job_id}_")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize job in database
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        try:
            cursor.execute('''
            INSERT INTO processing_jobs (job_id, status, start_time, total_files, processed_files)
            VALUES (%s, %s, %s, %s, %s)
            ''', (job_id, "processing", datetime.now(), 0, 0))
        except Exception as e:
            # If there's an issue with the table structure, try to fix it
            if "column \"error_message\" of relation \"processing_jobs\" does not exist" in str(e):
                # Add the missing column
                logger.info("Adding missing error_message column to processing_jobs table")
                cursor.execute('''
                ALTER TABLE processing_jobs 
                ADD COLUMN IF NOT EXISTS error_message TEXT
                ''')
                conn.commit()
                
                # Try the insert again
                cursor.execute('''
                INSERT INTO processing_jobs (job_id, status, start_time, total_files, processed_files)
                VALUES (%s, %s, %s, %s, %s)
                ''', (job_id, "processing", datetime.now(), 0, 0))
            else:
                raise e
        conn.commit()
        
        # Dictionary to store results for each folder
        folder_results = {}
        total_files = 0
        processed_files = 0
        
        # Process each zip file
        for zip_content, zip_name in zip(file_contents, file_names):
            
            # Extract the zip file to temp directory
            zip_dir = os.path.join(temp_dir, os.path.splitext(zip_name)[0])
            os.makedirs(zip_dir, exist_ok=True)
            
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                zf.extractall(zip_dir)
            
            # Process each folder in the zip
            for root, dirs, files in os.walk(zip_dir):
                # Skip the root directory
                if root == zip_dir:
                    continue
                
                # Get folder name (relative to zip)
                rel_path = os.path.relpath(root, zip_dir)
                folder_name = rel_path
                
                # Skip if there are no files
                if not files:
                    continue
                
                # Initialize folder results if not already present
                if folder_name not in folder_results:
                    folder_results[folder_name] = []
                
                # Track document types in this folder
                document_counts = Counter()
                
                # First pass: classify all documents
                for file in files:
                    if file.startswith('.') or file.startswith('~'):
                        continue  # Skip hidden files
                    
                    file_path = os.path.join(root, file)
                    total_files += 1
                    
                    try:
                        # Get file extension
                        _, ext = os.path.splitext(file)
                        
                        # Only process supported file types
                        if ext.lower() not in ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                            continue
                        
                        # Determine file type
                        file_type = {
                            '.pdf': 'application/pdf',
                            '.jpg': 'image/jpeg',
                            '.jpeg': 'image/jpeg',
                            '.png': 'image/png',
                            '.tiff': 'image/tiff',
                            '.tif': 'image/tiff'
                        }.get(ext.lower(), 'application/octet-stream')
                        
                        # Read file
                        with open(file_path, 'rb') as f:
                            file_data = f.read()
                        
                        # Perform OCR
                        ocr_result = DocumentProcessor.perform_ocr(file_data, file_type)
                        if "error" in ocr_result:
                            raise Exception(f"OCR failed: {ocr_result['error']}")
                        
                        # Classify document
                        classification = DocumentProcessor.classify_document(ocr_result["text"])
                        doc_type = classification["document_type"]
                        confidence = classification["confidence"]
                        
                        # Count document types
                        if doc_type != "unknown":
                            document_counts[doc_type] += 1
                            
                        # Store file info in database
                        cursor.execute('''
                        INSERT INTO processed_files 
                        (job_id, file_name, folder_name, file_path, document_type, confidence, processing_status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        ''', (
                            job_id, 
                            file, 
                            folder_name, 
                            file_path, 
                            doc_type, 
                            confidence, 
                            "classified"
                        ))
                        # Update total count immediately after each insertion
                        conn.commit()
                        
                        # Every 2 files, update the job status for better progress tracking
                        if total_files % 2 == 0:
                            cursor.execute('''
                            UPDATE processing_jobs 
                            SET total_files = %s, processed_files = %s 
                            WHERE job_id = %s
                            ''', (total_files, processed_files, job_id))
                            conn.commit()
                        
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {str(e)}")
                        # Record error in database
                        cursor.execute('''
                        INSERT INTO processed_files 
                        (job_id, file_name, folder_name, file_path, document_type, confidence, processing_status, error_message)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ''', (
                            job_id, 
                            file, 
                            folder_name, 
                            file_path, 
                            "unknown", 
                            0.0, 
                            "error", 
                            str(e)
                        ))
                        conn.commit()
                
                # Determine the dominant document type
                dominant_type = document_counts.most_common(1)
                if dominant_type:
                    dominant_doc_type = dominant_type[0][0]
                    logger.info(f"Folder {folder_name}: Dominant document type is {dominant_doc_type}")
                else:
                    dominant_doc_type = "unknown"
                    logger.warning(f"Folder {folder_name}: No valid documents found")
                
                # Second pass: process files that match the dominant type
                cursor.execute('''
                SELECT id, file_path, file_name FROM processed_files 
                WHERE job_id = %s AND folder_name = %s AND document_type = %s AND processing_status = 'classified'
                ''', (job_id, folder_name, dominant_doc_type))
                
                matching_files = cursor.fetchall()
                
                # Update job progress immediately after classifying files
                processed_files_count = len(matching_files)
                try:
                    cursor.execute('''
                    UPDATE processing_jobs 
                    SET total_files = %s, processed_files = %s 
                    WHERE job_id = %s
                    ''', (total_files, processed_files, job_id))
                    conn.commit()
                    logger.info(f"Updated job progress: {processed_files}/{total_files} files")
                except Exception as e:
                    logger.error(f"Error updating job progress after classification: {str(e)}")
                    # Try to reconnect if the connection was closed
                    try:
                        conn.close()
                    except:
                        pass
                    conn = get_db_connection()
                    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                
                for file_id, file_path, file_name in matching_files:
                    try:
                        processed_files += 1
                        
                        # Read file
                        with open(file_path, 'rb') as f:
                            file_data = f.read()
                        
                        # Get file extension
                        _, ext = os.path.splitext(file_path)
                        file_type = {
                            '.pdf': 'application/pdf',
                            '.jpg': 'image/jpeg',
                            '.jpeg': 'image/jpeg',
                            '.png': 'image/png',
                            '.tiff': 'image/tiff',
                            '.tif': 'image/tiff'
                        }.get(ext.lower(), 'application/octet-stream')
                        
                        # Perform OCR
                        ocr_result = DocumentProcessor.perform_ocr(file_data, file_type)
                        
                        # Extract fields
                        extraction_result = DocumentProcessor.extract_fields(ocr_result["text"], dominant_doc_type)
                        
                        # Store extracted fields
                        for field in extraction_result.get("extracted_fields", []):
                            field_name = field["field_name"]
                            value = field.get("value", "")
                            confidence = field.get("confidence", 0.0)
                            reason = field.get("reason", "")
                            
                            cursor.execute('''
                            INSERT INTO extracted_fields 
                            (job_id, file_id, field_name, field_value, confidence, reason)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ''', (job_id, file_id, field_name, value, confidence, reason))
                            
                            # Add to folder results
                            folder_results[folder_name].append({
                                "filepath": file_path,
                                "field_name": field_name,
                                "value": value,
                                "confidence": confidence,
                                "reason": reason
                            })
                        
                        # Update file status
                        cursor.execute('''
                        UPDATE processed_files 
                        SET processing_status = 'processed' 
                        WHERE id = %s
                        ''', (file_id,))
                        
                        conn.commit()
                        
                    except Exception as e:
                        logger.error(f"Error extracting fields from {file_path}: {str(e)}")
                        # Update file status
                        cursor.execute('''
                        UPDATE processed_files 
                        SET processing_status = 'error', error_message = %s 
                        WHERE id = %s
                        ''', (str(e), file_id))
                        conn.commit()
                
                # Mark files of other types as skipped but also include them in the results
                cursor.execute('''
                SELECT id, file_path, file_name, document_type FROM processed_files 
                WHERE job_id = %s AND folder_name = %s AND document_type != %s AND processing_status = 'classified'
                ''', (job_id, folder_name, dominant_doc_type))
                
                non_matching_files = cursor.fetchall()
                
                for file_id, file_path, file_name, doc_type in non_matching_files:
                    try:
                        # Update file status
                        cursor.execute('''
                        UPDATE processed_files 
                        SET processing_status = 'skipped', error_message = 'Document type does not match dominant type in folder'
                        WHERE id = %s
                        ''', (file_id,))
                        conn.commit()
                        
                        # Add to folder results with reason
                        folder_results[folder_name].append({
                            "filepath": file_path,
                            "document_type": doc_type,
                            "dominant_type": dominant_doc_type,
                            "processing_status": "skipped",
                            "reason": f"Document type '{doc_type}' does not match dominant type '{dominant_doc_type}' in folder"
                        })
                        
                    except Exception as e:
                        logger.error(f"Error marking non-matching file {file_path}: {str(e)}")

                
                # Update job progress
                try:
                    cursor.execute('''
                    UPDATE processing_jobs 
                    SET total_files = %s, processed_files = %s 
                    WHERE job_id = %s
                    ''', (total_files, processed_files, job_id))
                    conn.commit()
                except Exception as e:
                    logger.error(f"Error updating job progress: {str(e)}")
                    # Try to reconnect if the connection was closed
                    try:
                        conn.close()
                    except:
                        pass
                    conn = get_db_connection()
                    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                    cursor.execute('''
                    UPDATE processing_jobs 
                    SET total_files = %s, processed_files = %s 
                    WHERE job_id = %s
                    ''', (total_files, processed_files, job_id))
                    conn.commit()
        
        # Generate Excel report
        excel_path = os.path.join(output_dir, f"extraction_results_{job_id}.xlsx")
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            # Track all skipped documents for a separate sheet
            all_skipped_docs = []
            
            # Add all files and folders to the results if they're not already there
            cursor.execute('''
            SELECT pf.file_path, pf.folder_name, pf.document_type, pf.processing_status, pf.error_message
            FROM processed_files pf
            LEFT JOIN extracted_fields ef ON pf.id = ef.file_id
            WHERE pf.job_id = %s AND ef.id IS NULL
            ''', (job_id,))
            
            missing_files = cursor.fetchall()
            
            for missing_file in missing_files:
                file_path = missing_file["file_path"]
                folder_name = missing_file["folder_name"]
                doc_type = missing_file["document_type"]
                status = missing_file["processing_status"]
                error = missing_file["error_message"]
                
                # Initialize folder if not exists
                if folder_name not in folder_results:
                    folder_results[folder_name] = []
                
                # Add to results
                folder_results[folder_name].append({
                    "filepath": file_path,
                    "document_type": doc_type,
                    "processing_status": status,
                    "reason": error or "No fields extracted"
                })
            
            for folder_name, results in folder_results.items():
                if not results:
                    continue
                    
                # Create DataFrame
                df = pd.DataFrame()
                
                # Group by filepath
                filepath_groups = {}
                
                # Track documents with reasons
                folder_skipped_docs = []
                
                for item in results:
                    # Check if it's a document with a reason (different type)
                    if "processing_status" in item and item["processing_status"] in ["skipped", "error"]:
                        # Add folder name to the item for the all_skipped sheet
                        item["folder_name"] = folder_name
                        folder_skipped_docs.append(item)
                        all_skipped_docs.append(item)
                        continue
                        
                    filepath = item["filepath"]
                    if filepath not in filepath_groups:
                        filepath_groups[filepath] = {
                            "filepath": filepath,
                        }
                    
                    # Add field name, value, confidence, and reason
                    if "field_name" in item:
                        field_name = item["field_name"]
                        filepath_groups[filepath][field_name] = item["value"]
                        filepath_groups[filepath][f"{field_name}_conf"] = item["confidence"]
                        
                        if item.get("reason"):
                            filepath_groups[filepath][f"{field_name}_reason"] = item["reason"]
                
                # Convert to DataFrame
                if filepath_groups:
                    df = pd.DataFrame(list(filepath_groups.values()))
                    
                    # Reorder columns to put field and confidence side by side
                    cols = ["filepath"]
                    for field in FIELDS:
                        field_name = field["name"]
                        if field_name in df.columns:
                            cols.append(field_name)
                            cols.append(f"{field_name}_conf")
                            if f"{field_name}_reason" in df.columns:
                                cols.append(f"{field_name}_reason")
                    
                    # Use only columns that exist in the DataFrame
                    cols = [col for col in cols if col in df.columns]
                    if cols:  # Only reindex if columns exist
                        df = df[cols]
                
                # Create a sanitized sheet name (Excel has a 31 character limit for sheet names)
                sheet_name = re.sub(r'[\\/*?[\]:]', '_', folder_name)
                if len(sheet_name) > 31:
                    sheet_name = sheet_name[:28] + '...'
                
                # Create DataFrame for skipped docs in this folder
                if folder_skipped_docs and not df.empty:
                    df_skipped = pd.DataFrame(folder_skipped_docs)
                    df_skipped = df_skipped[["filepath", "document_type", "reason"]]
                    
                    # Label section
                    df_skipped_header = pd.DataFrame([{"filepath": "SKIPPED DOCUMENTS", "document_type": "", "reason": ""}])
                    
                    # Add an empty row as separator
                    df_empty = pd.DataFrame([{"filepath": "", "document_type": "", "reason": ""}])
                    
                    # Concatenate to include in the main sheet
                    df_all = pd.concat([df, df_empty, df_skipped_header, df_skipped], ignore_index=True)
                    df = df_all
                
                # For cases where there are only skipped docs in a folder
                elif folder_skipped_docs and df.empty:
                    df = pd.DataFrame(folder_skipped_docs)
                    # Select most relevant columns
                    display_cols = ["filepath", "document_type", "reason"]
                    display_cols = [col for col in display_cols if col in df.columns]
                    df = df[display_cols]
                
                # Write to Excel
                if not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Adjust column widths
                    worksheet = writer.sheets[sheet_name]
                    for i, col in enumerate(df.columns):
                        max_width = max(
                            df[col].astype(str).map(len).max(),
                            len(col)
                        ) + 2
                        worksheet.set_column(i, i, min(max_width, 50))  # Cap width at 50
            
            # Create a separate sheet for all skipped documents
            if all_skipped_docs:
                skipped_df = pd.DataFrame(all_skipped_docs)
                
                # Ensure the DataFrame has all important columns
                required_cols = ["filepath", "folder_name", "document_type", "reason"]
                for col in required_cols:
                    if col not in skipped_df.columns:
                        skipped_df[col] = None
                
                # Select and order columns (include only those that exist)
                display_cols = [col for col in required_cols if col in skipped_df.columns]
                skipped_df = skipped_df[display_cols]
                
                # Write to Excel
                skipped_df.to_excel(writer, sheet_name="Skipped_Documents", index=False)
                
                # Adjust column widths
                worksheet = writer.sheets["Skipped_Documents"]
                for i, col in enumerate(skipped_df.columns):
                    max_width = max(
                        skipped_df[col].astype(str).map(len).max(),
                        len(col)
                    ) + 2
                    worksheet.set_column(i, i, min(max_width, 50))  # Cap width at 50
        
        # Update job status and output file
        try:
            cursor.execute('''
            UPDATE processing_jobs 
            SET status = 'completed', end_time = %s, output_file_path = %s
            WHERE job_id = %s
            ''', (datetime.now(), excel_path, job_id))
        except Exception as e:
            # Check if error_message column exists, if not add it
            if "column \"error_message\" of relation \"processing_jobs\" does not exist" in str(e):
                # Add the missing column
                logger.info("Adding missing error_message column to processing_jobs table")
                cursor.execute('''
                ALTER TABLE processing_jobs 
                ADD COLUMN IF NOT EXISTS error_message TEXT
                ''')
                conn.commit()
                
                # Try the update again
                cursor.execute('''
                UPDATE processing_jobs 
                SET status = 'completed', end_time = %s, output_file_path = %s
                WHERE job_id = %s
                ''', (datetime.now(), excel_path, job_id))
            else:
                raise e
        conn.commit()
        
        cursor.close()
        conn.close()
        
        logger.info(f"Job {job_id} completed. Output file: {excel_path}")
        return excel_path
        
    except Exception as e:
        logger.error(f"Error processing zip files: {str(e)}")
        
        try:
            # Update job status
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            # Check if error_message column exists, if not add it
            try:
                cursor.execute('''
                UPDATE processing_jobs 
                SET status = 'failed', end_time = %s, error_message = %s
                WHERE job_id = %s
                ''', (datetime.now(), str(e), job_id))
            except Exception as column_error:
                if "column \"error_message\" of relation \"processing_jobs\" does not exist" in str(column_error):
                    # Add the missing column
                    logger.info("Adding missing error_message column to processing_jobs table")
                    cursor.execute('''
                    ALTER TABLE processing_jobs 
                    ADD COLUMN IF NOT EXISTS error_message TEXT
                    ''')
                    conn.commit()
                    
                    # Try the update again
                    cursor.execute('''
                    UPDATE processing_jobs 
                    SET status = 'failed', end_time = %s, error_message = %s
                    WHERE job_id = %s
                    ''', (datetime.now(), str(e), job_id))
                else:
                    raise column_error
                    
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as db_error:
            logger.error(f"Error updating job status: {str(db_error)}")
        
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as cleanup_error:
            logger.error(f"Error cleaning up temp directory: {str(cleanup_error)}")
        
        raise e
    finally:
        # Clean up temp directory after a delay to allow file download
        def delayed_cleanup():
            time.sleep(3600)  # Keep files for 1 hour
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.error(f"Error during delayed cleanup: {str(e)}")
        
        # Start cleanup in background
        import threading
        threading.Thread(target=delayed_cleanup).start()

@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Upload zip files containing folders of documents.
    Each zip file can contain multiple folders, and each folder will be analyzed separately.
    The dominant document type in each folder will be determined, and fields will be extracted accordingly.
    Results will be provided in an Excel file with one sheet per folder.
    """
    try:
        # Initialize database
        init_db()
        
        # Validate file types
        for file in files:
            if not file.filename.lower().endswith('.zip'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is not a zip file. Only zip files are supported."
                )
        
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Read the file contents into memory before background processing
        # This prevents "I/O operation on closed file" errors
        file_contents = []
        file_names = []
        for file in files:
            content = await file.read()
            file_contents.append(content)
            file_names.append(file.filename)
        
        # Start processing in the background with the file contents
        background_tasks.add_task(process_zip_files, file_contents, file_names, job_id)
        
        return {
            "status": "processing",
            "job_id": job_id,
            "message": "Files uploaded successfully. Processing started.",
            "files": file_names
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    try:
        logger.info("Starting status check.")
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        logger.info("Starting status check2.")

        cursor.execute('''
        SELECT * FROM processing_jobs WHERE job_id = %s
        ''', (job_id,))
        
        job = cursor.fetchone()
        logger.info("Starting status check3.")
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
        
        # Convert to dict
        job_dict = dict(job)
        
        # Convert datetime objects to strings
        for key in ["start_time", "end_time"]:
            if job_dict[key]:
                job_dict[key] = job_dict[key].isoformat()
        logger.info("Starting status check4.")
        
        # Get processing statistics
        cursor.execute('''
        SELECT COUNT(*) as total, processing_status, COUNT(*) FILTER (WHERE document_type != 'unknown') as recognized
        FROM processed_files 
        WHERE job_id = %s
        GROUP BY processing_status
        ''', (job_id,))
        
        stats = {}
        for row in cursor.fetchall():
            stats[row["processing_status"]] = {
                "count": row["total"],
                "recognized": row["recognized"]
            }
        
        job_dict["processing_stats"] = stats

        logger.info("Starting status check5.")
        
        cursor.close()
        conn.close()
        
        return job_dict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
   
@app.get("/download/{job_id}")
async def download_results(job_id: str):
    """Download the results of a completed job."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        cursor.execute('''
        SELECT status, output_file_path FROM processing_jobs WHERE job_id = %s
        ''', (job_id,))
        
        job = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
        
        if job["status"] != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Job is not completed. Current status: {job['status']}"
            )
        
        output_path = job["output_file_path"]
        
        if not output_path or not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail="Output file not found")
        
        return FileResponse(
            path=output_path,
            filename=f"extraction_results_{job_id}.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

if __name__ == "__main__":
    # Initialize database
    init_db()
    # Start the FastAPI server
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)