import os
from pymongo import MongoClient


def main() -> None:
    mongodb_uri = os.getenv("MONGODB_URI") or (
        "mongodb+srv://harilogicgo_db_user:g6Zz4M2xWpr3B2VM@cluster0.bnzjt7f.mongodb.net/"
        "?retryWrites=true&w=majority&appName=Cluster0"
    )
    only = {"skull_dress.jpg", "witch_dress.webp", "vampire_dress.jpg"}

    client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    db = client.halloween_db
    garments = db.garments

    # Ensure presence with correct URL
    for fn in only:
        garments.update_one(
            {"filename": fn},
            {"$set": {"filename": fn, "url": f"/garment_templates/{fn}"}},
            upsert=True,
        )

    res = garments.delete_many({"filename": {"$nin": list(only)}})
    print({"deleted": res.deleted_count})


if __name__ == "__main__":
    main()


