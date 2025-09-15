# Real-Time Financial Sentiment Analysis for YouTube Comments

This project provides a real-time sentiment analysis of comments on financial YouTube videos through a Chrome extension. It leverages a machine learning model hosted on AWS to deliver instant insights, including sentiment distribution, trend graphs, and word clouds, directly within the user's browser.

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project is a full-stack MLOps system for analyzing public sentiment on youtube. It began with an in-depth exploratory data analysis (EDA) and intense training of a sentiment classification model. The finalized model was then deployed through an automated pipeline in to a scalable, cloud-native API. A custom Chrome extension serves as the frontend, calling this API to provide users with immediate sentiment visualization for any youtube video, creating a powerful tool for market research and public opinion analysis.

---

## Architecture

The system is designed with a decoupled frontend and backend architecture, containerized for portability and deployed on a serverless AWS infrastructure.


**Workflow:**
1.  The **Chrome Extension** is activated on a YouTube video page.
2.  It uses the **YouTube Data API v3** to fetch the latest comments.
3.  The comments are sent to a **Flask API endpoint** hosted on AWS.
4.  The request is handled by an **Application Load Balancer (ALB)**, which routes traffic to the backend.
5.  The backend is a **Docker container** managed by **Amazon ECS with Fargate**, ensuring scalability and removing the need for server management.
6.  The Flask application uses a pre-trained **Scikit-learn model** to perform sentiment analysis.
7.  The results, including generated visualizations (charts, word clouds), are sent back to the Chrome extension.
8.  The extension dynamically renders the insights in the popup.

---

## Features

* **Real-Time Analysis:** Get instant sentiment feedback on YouTube comments.
* **Comprehensive Metrics:** View total comments, unique commenters, and average comment length.
* **Sentiment Score:** A normalized sentiment score from 0 to 10.
* **Sentiment Distribution:** A pie chart showing the percentage of positive, neutral, and negative comments.
* **Sentiment Trend:** A graph illustrating how sentiment has changed over time.
* **Word Cloud:** A visual representation of the most frequently used words in the comments.
* **Top Comments:** A list of the top comments with their predicted sentiment.

---

## Tech Stack

* **Frontend:** JavaScript, HTML, CSS, Chrome Extension APIs
* **Backend:** Python, Flask, Gunicorn
* **ML / Data Science:** Scikit-learn, Pandas, NLTK, Tensorflow, GLove, Word2vec, Matplotlib, WordCloud,
* **MLOps & Deployment:** Docker,MLFlOw,DVC, Amazon ECR (Elastic Container Registry), Amazon ECS (Elastic Container Service), AWS Fargate, Application Load Balancer (ALB)

---

## Project Structure

├── Datasets
│   ├── Embeddings
│   └── Financial_data.csv
├── Flask_app
│   └── app.py
├── Notebooks
│   ├── DT_BOW.ipynb
│   ├── Navie_bayes.ipynb
│   ├── Sentiment_LSTM_Att.ipynb
│   ├── classification_report.csv
│   ├── confusion_matrix_Bag of Words.png
│   ├── confusion_matrix_TF-IDF.png
│   ├── dataIngestion_preprocessing.ipynb
│   ├── download_embeddings.py
│   ├── ml_models.ipynb
│   ├── ml_models_dataaug.ipynb
├── README.md
├── bow_vectorizer.pkl   # the final trained vector artifact
├── decisiontree_model.pkl  # the final trained model artifact
├── dockerfile
├── dvc.yaml
├── params.yaml
├── requirement.txt
├── requirement_core.txt
├── requirement_dl.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── data
│   │   ├── data_preprocessing.py
│   │   └── make_dataset.py
│   └── model
│       ├── model_building.py
│       ├── model_evaluation.py
│       └── register_model.py
└── yt_chrome_plugin
    ├── Manifest.json
    ├── popup.html
    └── popup.js

.
├── Dockerfile              # Recipe for building the production Docker image
├── Flask_app/              # Contains the Flask backend for serving the model
│   └── app.py
├── Notebooks/              # Jupyter notebooks for experimentation and analysis
│   ├── dataIngestion_preprocessing.ipynb
│   └── ml_models.ipynb
├── Datasets/               # Raw and processed data
│   └── Financial_data.csv
├── src/                    # Source code for the installable Python package
│   ├── data/
│   ├── model/
│   └── __init__.py
├── yt_chrome_plugin/       # Files for the Chrome extension frontend
│   ├── Manifest.json
│   ├── popup.html
│   └── popup.js
├── requirements.txt        # Python dependencies for the project
├── setup.py                # Makes the project an installable package
├── decisiontree_model.pkl  # The final trained model artifact
└── bow_vectorizer.pkl      # The final trained vectorizer artifact

## Local Setup and Installation

To run this project on your local machine:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/therealLakshman/Financial_sentiment_prediction.git](https://github.com/therealLakshman/Financial_sentiment_prediction.git)
    cd Financial_sentiment_prediction
    ```
2.  **Set Up Virtual Environment:**
    ```bash
    python -m venv fin_ops
    source fin_ops/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirement_core.txt
    ```
4.  **Run the Backend Server:**
    ```bash
    python Flask_app/app.py
    ```
    The server will be running at `http://localhost:5000`.

5.  **Load the Chrome Extension:**
    * Open Chrome and navigate to `chrome://extensions/`.
    * Enable "Developer mode".
    * Click "Load unpacked" and select the `yt_chrome_plugin` directory.
    * Ensure the `API_URL` in `popup.js` is set to `http://localhost:5000/.
## Deployment (MLOps Pipeline on AWS)

This project is deployed as a containerized application on a serverless AWS infrastructure.

1.  **Containerization:**
    * A `Dockerfile` is used to package the Flask application, the trained model files, and all dependencies into a portable Docker image.
    * The image is built locally using:
        ```bash
        docker build -t financial-sentiment-backend .
        ```

2.  **Pushing to Registry (ECR):**
    * Amazon ECR is used as a private registry to store the Docker image.
    * The image is tagged and pushed to ECR using the AWS CLI and Docker commands.
        ```bash
        # Log in to ECR
        aws ecr get-login-password ... | docker login ...

        # Tag the image
        docker tag financial-sentiment-backend:latest <your_ecr_uri>

        # Push the image
        docker push <your_ecr_uri>
        ```

3.  **Deployment (ECS with Fargate):**
    * **Task Definition:** A blueprint is created in ECS to define the container's specifications (image URL from ECR, CPU/memory, port mappings).
    * **Cluster:** A serverless cluster using the "Networking only" template is created to host the application.
    * **Service:** An ECS Service is created to run and maintain the desired number of tasks (containers). This service is configured to use an **Application Load Balancer (ALB)** to distribute traffic.

4.  **Networking & Security:**
    * The ALB's **Security Group** is configured with an **inbound rule** to allow HTTP traffic on port 80 from the internet (`0.0.0.0/0`).
    * The ALB's **Listener** is configured for port 80, and it forwards traffic to a **Target Group** pointing to the container's internal port `5000`.

5.  **Final Frontend Update:**
    * The `API_URL` constant in `popup.js` is updated with the public DNS name of the Application Load Balancer.
    * The Chrome extension is repackaged and published.

---

## Usage

1.  Navigate to any financial analysis video on YouTube.
2.  Click the extension icon in the Chrome toolbar.
3.  The popup will display a complete sentiment analysis of the video's comments.

