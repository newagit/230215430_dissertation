from pymongo import MongoClient
import hashlib

# MongoDB Setup
client = MongoClient("mongodb://localhost:27017/")
db = client['crime_branch']
users_collection = db['users']
data_collection = db['data']

# Utility function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Seeding Users Collection
admin_user = {
    "username": "admin",
    "password": hash_password("admin123")  # Change as needed
}

# regular_user = {
#     "username": "user",
#     "password": hash_password("user123")  # Change as needed
# }

# Insert users if they don't already exist
if users_collection.count_documents({"username": "admin"}) == 0:
    users_collection.insert_one(admin_user)
    print("Admin user added.")

if users_collection.count_documents({"username": "user"}) == 0:
    users_collection.insert_one(regular_user)
    print("Regular user added.")

# Seeding Crime Data Collection
dummy_data = [
    {"field1": "John Doe", "field2": "Theft"},
    {"field1": "Jane Smith", "field2": "Fraud"},
    {"field1": "Mike Johnson", "field2": "Burglary"}
]

if data_collection.count_documents({}) == 0:
    data_collection.insert_many(dummy_data)
    print("Crime data seeded successfully.")
else:
    print("Crime data already exists.")

print("Seeding completed!")
