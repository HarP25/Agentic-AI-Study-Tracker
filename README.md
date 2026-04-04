# 🧠 AI Study Agent

> AI-powered study tracker that adapts schedules, analyzes focus patterns, and optimizes learning using intelligent agents.

---

## ⚡ Overview

AI Study Agent is a backend-driven system that tracks user study behavior, predicts focus trends, and dynamically adjusts study schedules.

Instead of static timetables, this system uses **agent-based logic** to create a personalized learning experience.

---

## 🚀 Features

- 📊 Track study sessions and behavior
- 🧠 AI-powered focus pattern analysis
- 🔄 Dynamic schedule adaptation
- ⚙️ Backend-driven decision system
- 🌐 Simple web interface for interaction

---

## 🏗️ Tech Stack

- **Backend:** Python (Flask)
- **Frontend:** HTML, CSS, JavaScript
- **AI Integration:** Groq API
- **Deployment:** Render

---

## 📂 Project Structure

project/
│── app.py
│── templates/
│ └── index.html
│── static/
│── requirements.txt
│── .env.example
│── LICENSE

## 🔐 Environment Variables

Create a `.env` file in the root directory:


SECRET_KEY=your_secret_key_here
GROQ_API_KEY=your_api_key_here
TOKEN_EXPIRY_HOURS=24
DB_PATH=study_agent.db


---

## 🛠️ Setup & Run Locally

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo

pip install -r requirements.txt

python app.py
