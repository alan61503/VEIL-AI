from cloud.firebase_client import init_firebase


if __name__ == "__main__":
	db = init_firebase()
	print("Firebase connected:", db is not None)
