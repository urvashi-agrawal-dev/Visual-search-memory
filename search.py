import os
import pytesseract
from PIL import Image
import json
SCREENSHOT_FOLDER="screenshots"
OUTPUT_FILE="results.json"
def extract_text_from_images():
    data=[]
    for filename in os.listdir(SCREENSHOT_FOLDER):
        if filename.lower().endswith((".png",".jpg",".jpeg")):
            filepath=os.path.join(SCREENSHOT_FOLDER,filename)
            image=Image.open(filepath)
            text=pytesseract.image_to_string(image)
            data.append({
                "filename":filename,
                "text":text.strip()
            })
            print(f"Extracted text from {filename}")
    with open (OUTPUT_FILE,"w",encoding="utf-8") as f:
        json.dump(data,f,indent=2)
    print("All text saved to results.json")
def search_query(query):
    with open(OUTPUT_FILE,"r",encoding="utf-8") as f:
        data=json.load(f)
    results=[]
    for item in data:
        if query.lower() in item["text"].lower():
            results.append(item["filename"])
    if results:
        print("FOund matches in:")
        for r in results:
            print("-",r)
    else:
        print("No matches found")
if __name__=="__main__":
    extract_text_from_images()
    query=input("Enter your search query:")
    search_query(query)