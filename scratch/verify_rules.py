import requests
import json
import os

url = "http://localhost:5000/predict"

def test_file(filepath):
    print(f"\n--- Testing file: {filepath} ---")
    if not os.path.exists(filepath):
        print(f"File {filepath} not found!")
        return

    with open(filepath, 'rb') as f:
        files = {'file': (os.path.basename(filepath), f, 'image/jpeg')}
        data = {'threshold': '0.5'}
        response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        res_data = response.json()
        print("Status: 200 OK")
        print(f"Label: {res_data.get('label')}")
        print(f"Confidence: {res_data.get('confidence')}")
        print(f"Grade: {res_data.get('grade')}")
        print(f"Reliability Reduced: {res_data.get('reliability_reduced')}")
        print(f"OOD Detected: {res_data.get('ood_detected')}")
        print(f"Insufficient Mask Data: {res_data.get('insufficient_mask_data')}")
        print(f"Alert level: {res_data.get('alert', {}).get('level')}")
        print(f"Alert title: {res_data.get('alert', {}).get('title')}")
        print(f"Alert message: {res_data.get('alert', {}).get('message')}")
        print(f"Pipeline status block: {json.dumps(res_data.get('pipeline_status'), indent=2)}")
        print(f"Measurements: {json.dumps(res_data.get('measurements'), indent=2)}")
    else:
        print(f"Status: {response.status_code}")
        print(response.text)

# Test healthy scan (Te-noTr_0000.jpg)
test_file("d:/DebobratoResearch/BrainTumorDetection/dummy_data/Testing/notumor/Te-noTr_0000.jpg")

# Test tumor scan (img1.png)
test_file("d:/DebobratoResearch/BrainTumorDetection/dummy_data/tumor/img1.png")
