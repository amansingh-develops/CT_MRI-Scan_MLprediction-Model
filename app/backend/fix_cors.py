import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

# Path to the service account key
cred_path = r"c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\app\backend\scansight-339d2-firebase-adminsdk-fbsvc-cdd77ee975.json"

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'scansight-339d2.firebasestorage.app'
})

bucket = storage.bucket()
bucket.cors = [
    {
        "origin": ["*"],
        "method": ["GET", "PUT", "POST", "DELETE", "OPTIONS"],
        "responseHeader": ["*"],
        "maxAgeSeconds": 3600
    }
]
bucket.patch()
print("CORS configuration updated successfully for Firebase Storage!")
