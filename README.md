# CampusRadar Internship Recommendation System

## Overview

The Internship Recommendation System helps connect students with suitable internships and assists recruiters in finding the most relevant candidates. It uses semantic search and Large Language Models (LLMs) to improve the recommendation process.

The system provides two main functionalities:

* Recommend the most suitable candidates for an internship.
* Recommend the most relevant internships for a student.

---

## Tech Stack

* FastAPI
* LangChain
* OpenAI
* ChromaDB
* Pydantic

---

## Features

* Store internship and candidate profiles in ChromaDB.
* Generate embeddings using OpenAI Embeddings.
* Perform semantic similarity search using cosine similarity.
* AI-based ranking of candidates using an LLM.
* Clean layered FastAPI architecture.
* Request validation using Pydantic.

---

## API Endpoints

### POST `/ingest`

Stores internship or candidate profiles in ChromaDB.

### POST `/recommend/candidates`

Receives an internship description and a list of applicant profiles, then returns the recommended candidate IDs.

### POST `/recommend/internships`

Receives a student's profile and returns the most relevant internship postings using semantic similarity search.

---

## How It Works

### Candidate Recommendation

* Receive the internship description.
* Receive the applicant profiles.
* Use the LLM to compare all applicants with the internship requirements.
* Return the recommended candidate IDs.

### Internship Recommendation

* Receive the student's profile.
* Generate an embedding using OpenAI Embeddings.
* Perform cosine similarity search in ChromaDB.
* Return the most relevant internship postings.

---

**Vaibhavi Kadam**
