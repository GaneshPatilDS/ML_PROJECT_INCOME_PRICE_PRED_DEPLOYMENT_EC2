from pymongo.mongo_client import MongoClient
import pandas as pd
import json

uri = "mongodb+srv://patil:patil@cluster0.zldquum.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


# create database name and collection name
DATABASE_NAME="INCOME"
COLLECTION_NAME="DATA"

df = pd.read_csv('notebooks/data/adult.data',names = ['age', 'workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','income'])
print(df)



# Convert the data into json
json_record=list(json.loads(df.T.to_json()).values())

#now dump the data into the database
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)