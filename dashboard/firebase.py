import firebase_admin
from firebase_admin import credentials, auth, firestore
import json
import os

creds_json = os.getenv("GOOGLE_CREDENTIALS")
creds_dict = json.loads(creds_json)

# Path to your Firebase Admin SDK JSON file
cred = credentials.Certificate(creds_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()