import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import os

# Path to the service account key
SERVICE_ACCOUNT_PATH = r"c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\app\backend\scansight-339d2-firebase-adminsdk-fbsvc-cdd77ee975.json"

def main():
    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        print(f"Service account file not found: {SERVICE_ACCOUNT_PATH}")
        return

    # Initialize Firebase Admin SDK
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'scansight-339d2.firebasestorage.app'
    })

    # Get a reference to the storage service, which gives us a Google Cloud Storage Bucket object
    bucket = storage.bucket()

    # Define the CORS configuration
    cors_configuration = [
        {
            "origin": ["*"],
            "method": ["GET", "HEAD", "PUT", "POST", "DELETE", "OPTIONS"],
            "responseHeader": ["Content-Type", "Authorization", "Content-Length", "User-Agent", "x-goog-resumable"],
            "maxAgeSeconds": 3600
        }
    ]

    # Apply the CORS configuration
    bucket.cors = cors_configuration
    bucket.patch()

    print(f"Successfully updated CORS configuration for bucket: {bucket.name}")
    print("Refresh your React app and try uploading again!")

if __name__ == "__main__":
    main()
