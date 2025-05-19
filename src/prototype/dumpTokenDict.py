import pickle

inpath = "/home/tumlinson/GlassHelix/Dataset/token_dictionary.pkl"

outpath = "/home/tumlinson/GlassHelix/Dataset/token_dictionary.csv"

with open(inpath, "rb") as f:
    token_dict = pickle.load(f)

# Convert the dictionary to a CSV-like format
with open(outpath, "wb") as f:
    for key, value in token_dict.items():
        # Write the key and value to the file
        f.write(f"{key},{value}\n".encode())
        