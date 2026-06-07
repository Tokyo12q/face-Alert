# ─────────────────────────────────────────────────────────────────────────────
# download_models.py
# Purpose: Download OpenCV Haar Cascade XML model files from GitHub and save
#          them into a local "models" folder so the face-detection system can
#          use them offline.
# ─────────────────────────────────────────────────────────────────────────────

import os        # Built-in Python module for interacting with the operating
                 # system (e.g. checking if a folder exists, creating folders,
                 # building file paths).

import requests  # Third-party library used to make HTTP network requests
                 # (e.g. downloading files from a URL over the internet).


# ─── Helper Function ──────────────────────────────────────────────────────────

def download_file(url, folder):
    """Download a single file from 'url' and save it inside 'folder'."""

    # Check whether the destination folder already exists on disk.
    # os.path.exists() returns True if the path is found, False otherwise.
    if not os.path.exists(folder):

        # If the folder does NOT exist, create it (and any missing parent
        # directories).  'makedirs' is like running "mkdir -p" on Linux.
        os.makedirs(folder)

    # Extract just the file name from the full URL.
    # url.split('/') splits the URL string on every '/' character into a list.
    # [-1] selects the LAST element of that list, which is the file name.
    # Example: "https://.../haarcascade_frontalface_default.xml"  →
    #          "haarcascade_frontalface_default.xml"
    filename = url.split('/')[-1]

    # Build the full local path where the file will be saved.
    # os.path.join() combines the folder name and file name with the correct
    # path separator for the current OS (\ on Windows, / on Linux/Mac).
    # Example: "models" + "haarcascade_frontalface_default.xml"
    #          → "models\haarcascade_frontalface_default.xml"
    filepath = os.path.join(folder, filename)

    # Print a progress message to the console so the user knows which file
    # is currently being downloaded.  f-strings let us embed 'filename'
    # directly inside the string using curly braces {}.
    print(f"Downloading {filename}...")

    # Send an HTTP GET request to the URL (like opening the URL in a browser).
    # The server's full response (headers + body/content) is stored in
    # 'response'.
    response = requests.get(url)

    # HTTP status code 200 means "OK" – the request succeeded and the server
    # returned the file content.
    if response.status_code == 200:

        # Open (or create) the local file at 'filepath' in write-binary mode
        # ('wb').  Binary mode is required for non-text files like XML models.
        # The 'with' statement automatically closes the file when the block
        # finishes, even if an error occurs.
        with open(filepath, 'wb') as f:

            # Write the raw bytes of the downloaded file into the local file.
            # response.content holds the response body as a bytes object.
            f.write(response.content)

        # Confirm to the user that the file was saved successfully.
        print(f"Saved to {filepath}")

    else:
        # If the status code is anything other than 200 (e.g. 404 Not Found,
        # 403 Forbidden), notify the user that the download failed.
        print(f"Failed to download {filename}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

# This block only runs when the script is executed directly (e.g.
# "python download_models.py").  It does NOT run if this file is imported
# as a module by another script.
if __name__ == "__main__":

    # The base URL pointing to OpenCV's official GitHub repository where the
    # pre-trained Haar Cascade XML files are stored.
    base_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"

    # A list of the specific model file names we want to download:
    #   1. haarcascade_frontalface_default.xml – detects frontal human faces.
    #   2. haarcascade_eye.xml                 – detects eyes inside a face.
    models = [
        "haarcascade_frontalface_default.xml",  # Face detection model
        "haarcascade_eye.xml"                   # Eye detection model
    ]

    # Loop over each model name in the list one by one.
    # On each iteration, 'model' holds the current file name string.
    for model in models:

        # Concatenate the base URL with the model file name to form the full
        # download URL, then call download_file() to fetch and save it.
        # The second argument "models" is the local folder to save the file in.
        # Example full URL:
        #   "https://raw.githubusercontent.com/.../haarcascade_frontalface_default.xml"
        download_file(base_url + model, "models")
