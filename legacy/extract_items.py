import ijson
from tqdm import tqdm

def extract_keys(filename):
    keys = set()
    with open(filename, "rb") as f:
        for prefix, event, value in tqdm(ijson.parse(f), desc="Parsing JSON", unit="event"):
            if event == "map_key":
                keys.add(value)
    return keys

def extract_key_paths(filename):
    paths = set()
    with open(filename, "rb") as f:
        for prefix, event, value in tqdm(ijson.parse(f), desc="Parsing JSON", unit="event"):
            if event == "map_key":
                path = f"{prefix}.{value}" if prefix else value
                paths.add(path)
    return paths

if __name__ == "__main__":
    keys = extract_keys("./data_processed/tweet_0_processed.json")
    print("Chiavi uniche:", keys)

    paths = extract_key_paths("./data_processed/tweet_0_processed.json")
    print("Percorsi chiave:", paths)
