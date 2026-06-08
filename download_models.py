import os

import requests

def download_file(url, folder):

    if not os.path.exists(folder):

        os.makedirs(folder)

    filename = url.split('/')[-1]

    filepath = os.path.join(folder, filename)

    print(f"Downloading {filename}...")

    response = requests.get(url)

    if response.status_code == 200:

        with open(filepath, 'wb') as f:

            f.write(response.content)

        print(f"Saved to {filepath}")

    else:
        print(f"Failed to download {filename}")

if __name__ == "__main__":

    base_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"

    models = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_eye.xml"
    ]

    for model in models:

        download_file(base_url + model, "models")
