from motor import motor_asyncio
from core.config import settings


class Database:
    def __init__(self):
        self.client = motor_asyncio.AsyncIOMotorClient(settings.DATABASE_URL)
        self.db = self.client[settings.MONGO_DATABASE_NAME]

    def insert_one(self, collection, data):
        return self.db[collection].insert_one(data)

    def find_one(self, collection, data):
        return self.db[collection].find_one(data)

    def find(self, collection, data):
        return self.db[collection].find(data, {"_id": 0})

    def update_one(self, collection, data, new_data):
        return self.db[collection].update_one(data, new_data)

    def delete_one(self, collection, data):
        return self.db[collection].delete_one(data)

    def delete_many(self, collection, data):
        return self.db[collection].delete_many(data)

    def drop_collection(self, collection):
        return self.db[collection].drop()

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(Database, cls).__new__(cls)
        return cls.instance


db = Database()