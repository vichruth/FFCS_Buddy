# FFCS_Buddy
This is a project created during a hackathon event.
FFCS Buddy 🚀

An AI-powered course and faculty recommender built to bring sanity to VIT's Fully Flexible Credit System (FFCS). This project was developed for the Hackademia - College Survival Edition hackathon.

🎯 The Problem

Every semester, students at VIT face the chaotic and stressful process of course registration through FFCS. Choosing the right courses with the best professors and a convenient timetable from hundreds of options is overwhelming. Students often rely on fragmented and biased advice from seniors, leading to poor choices and a difficult semester.

✨ What It Does

FFCS Buddy is an intelligent assistant that helps students make informed decisions during course registration. It provides a smart search and recommendation engine with multiple modes of interaction:

    🏷️ Keyword Filtering: Students can select specific tags (e.g., "project-based", "chill", "lenient-grading") to filter for professors who match their exact preferences.

    🧠 Semantic Search: Students can type natural language queries like "a relaxed professor who gives hands-on assignments". The AI understands the meaning behind the query and finds the best conceptual matches, even if the exact keywords aren't present.

    🔧 Hybrid Search: The most powerful feature, combining both filters and semantic search. A student can filter for all "CSE1004" courses in the "A1" slot and then rank the available professors based on who is most similar to "a teacher that gives good notes".

🛠️ Tech Stack

This project's backend is built with a modern Python stack focused on performance and machine learning.

    Backend Framework: FastAPI

    ML Server: Uvicorn

    Core ML Libraries:

        scikit-learn: For the efficient keyword-based filtering model.

        sentence-transformers (PyTorch): For the state-of-the-art semantic search model (all-MiniLM-L6-v2).

        pandas: For data manipulation.

    Data: A synthetically generated dataset of over 1000 faculty members and course offerings to simulate the real FFCS environment.

⚙️ System Architecture

The application follows a simple and robust client-server architecture:

Frontend (Client) ➡️ FastAPI (Backend API) ➡️ ML Engine (engine.py)

    The user interacts with the front end.

    The front end sends a JSON request to the FastAPI backend.

    The FastAPI server calls the appropriate function in the ML engine.

    The engine performs the filtering and/or ranking and returns the results.

    FastAPI sends the results back to the front end as a JSON response.

🚀 Getting Started

Follow these steps to run the backend server on your local machine.

Prerequisites

    Python 3.9+

    Conda package manager

Installation & Setup

    Clone the repository:
    Bash

git clone https://github.com/your-username/FFCS_Buddy.git
cd FFCS_Buddy

Create and activate the Conda environment:
Bash

conda create --name ffcs_env python=3.11
conda activate ffcs_env

Create your requirements.txt file:
This is a very important step. Run the following command in your terminal to save all your project's libraries.
Bash

pip freeze > requirements.txt

Install the dependencies:
Bash

    pip install -r requirements.txt

Running the Server

To start the backend API server, run the following command. The --host 0.0.0.0 flag makes it accessible to other devices on your local network (like your front-end team's computers).
Bash

uvicorn main:app --reload --host 0.0.0.0

The server will be running at http://YOUR_LOCAL_IP:8000.

📡 API Endpoint

The primary endpoint for all recommendations.

POST /find-courses

This endpoint handles keyword, semantic, and hybrid searches.

Request Body Example (Hybrid Search):
JSON

{
  "course_code": "CSE1004",
  "slot": "A1",
  "query": "an engaging teacher",
  "style_prefs": ["project-based"],
  "top_n": 5
}

Response Body Example:
JSON

{
  "status": "success",
  "results": [
    {
      "faculty_name": "Dr. Anand R.",
      "department": "SCOPE",
      "rating": 4.3,
      "style_tags": "project-based,chill,good-notes",
      "similarity_score": 0.685
    }
  ]
}

🔮 Future Improvements

    Integrate with a real database to store and retrieve data.

    Add a user review and rating submission feature.

    Implement a full timetable generation feature to avoid course clashes.

    Scrape real, live FFCS data to provide up-to-date recommendations.
