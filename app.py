import os
import re
import json
import io
import concurrent.futures
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
import numpy as np, requests
import datamapplot, arxiv, openai
from requests.adapters import HTTPAdapter
from difflib import SequenceMatcher
from bs4 import BeautifulSoup
import random  # For extra recommendations

app = Flask(__name__)

# Load OpenAI API key from environment variable.
openai.api_key = os.getenv("OPENAI_API_KEY")

# In-memory cache for paper details.
paper_cache = {}

# Global cache for latest papers, to persist for the current day.
latest_papers_cache = {
    "date": None,
    "papers": []
}

def fetch_url(url):
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=3)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})

def create_plot():
    base_url = "https://github.com/TutteInstitute/datamapplot"
    data_map_file = fetch_url(f"{base_url}/raw/main/examples/arxiv_ml_data_map.npy")
    arxivml_data_map = np.load(io.BytesIO(data_map_file.content))
    arxivml_label_layers = []
    for layer_num in range(5):
        label_file = fetch_url(f"{base_url}/raw/interactive/examples/arxiv_ml_layer{layer_num}_cluster_labels.npy")
        arxivml_label_layers.append(np.load(io.BytesIO(label_file.content), allow_pickle=True))
    hover_data_file = fetch_url(f"{base_url}/raw/interactive/examples/arxiv_ml_hover_data.npy")
    arxiv_hover_data = np.load(io.BytesIO(hover_data_file.content), allow_pickle=True)
    # Use a modern font for the graph
    plot = datamapplot.create_interactive_plot(
        arxivml_data_map,
        arxivml_label_layers[0],
        arxivml_label_layers[2],
        arxivml_label_layers[4],
        hover_text=arxiv_hover_data,
        font_family="Raleway",
        on_click="window.parent.fetchPaperDetails(`{hover_text}`);",
        enable_search=True,
        darkmode=False,
        initial_zoom_fraction=0.5,
    )
    return str(plot)

@app.route('/plot')
def plot_view():
    return create_plot()

@app.route('/')
def landing():
    return render_template("landing.html")

@app.route('/app')
def main_app():
    return render_template("index.html")

def search_variant(query, title):
    try:
        search = arxiv.Search(
            query=query,
            max_results=20,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        candidates = list(search.results())
        best_candidate = None
        best_ratio = 0.0
        for candidate in candidates:
            ratio = SequenceMatcher(None, title.lower(), candidate.title.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_candidate = candidate
        return best_candidate, best_ratio
    except Exception:
        return None, 0.0

def query_arxiv_by_title_fast(title):
    # Try a looser query based on a partial title first.
    words = title.split()
    if len(words) > 3:
        partial_title = " ".join(words[:min(5, len(words))])
        looser_query = f'all:"{partial_title}"'
    else:
        looser_query = title
    candidate, ratio = search_variant(looser_query, title)
    if candidate is not None:
        return candidate

    # Fallback to stricter queries concurrently.
    query_variants = [
        f'ti:"{title}"',
        f'all:"{title}"',
        f'ti:{title}',
        title
    ]
    best_candidate = None
    best_ratio = 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(query_variants)) as executor:
        futures = {executor.submit(search_variant, query, title): query for query in query_variants}
        for future in concurrent.futures.as_completed(futures):
            cand, rat = future.result()
            if cand is not None and rat > best_ratio:
                best_ratio = rat
                best_candidate = cand
    return best_candidate

@app.route('/get_arxiv_details')
def get_arxiv_details():
    title = request.args.get('title', '')
    if not title:
        return jsonify({"error": "No title provided"}), 400
    if title in paper_cache:
        return jsonify(paper_cache[title])
    try:
        result = query_arxiv_by_title_fast(title)
    except Exception as e:
        return jsonify({"error": "Error fetching results: " + str(e)}), 500
    if result is None:
        return jsonify({"error": f"Cannot extract paper details for '{title}'."})
    details = {
        "id": result.entry_id,
        "updated": result.updated.isoformat() if result.updated else "",
        "published": result.published.isoformat() if result.published else "",
        "title": result.title,
        "abstract": result.summary,
        "authors": ", ".join([author.name for author in result.authors]) if result.authors else "",
        "comment": result.comment or "",
        "journal_ref": result.journal_ref or "",
        "doi": result.doi or "",
        "primary_category": result.primary_category,
        "categories": ", ".join(result.categories) if result.categories else "",
        "alternate_link": next((link.href for link in result.links if link.rel == "alternate"), ""),
        "pdf_link": result.pdf_url or next((link.href for link in result.links if link.title == "pdf"), "")
    }
    paper_cache[title] = details
    return jsonify(details)

@app.route('/ask_paper', methods=['POST'])
def ask_paper():
    data = request.get_json()
    question = data.get("question")
    title = data.get("title")
    if not question or not title:
        return jsonify({"error": "Missing 'question' or 'title' in payload."}), 400
    result = query_arxiv_by_title_fast(title)
    if result is None:
        paper_details = {"error": f"Cannot extract paper details for '{title}'."}
        return jsonify({"response": paper_details["error"], "paper_details": paper_details})
    else:
        paper_details = {
            "title": result.title,
            "authors": ", ".join([author.name for author in result.authors]) if result.authors else "",
            "doi": result.doi or "",
            "abstract": result.summary,
            "published": result.published.isoformat() if result.published else ""
        }
    let_context = "Title: " + paper_details["title"]
    if paper_details["authors"]:
        let_context += "\nAuthors: " + paper_details["authors"]
    if paper_details["doi"]:
        let_context += "\nDOI: " + paper_details["doi"]
    if paper_details["abstract"]:
        let_context += "\nAbstract: " + paper_details["abstract"]
    if paper_details["published"]:
        let_context += "\nPublished: " + paper_details["published"]
    prompt = (
        "Using the paper details provided below as context, answer the following research question. "
        "For every sentence in your answer, include an inline citation formatted as a numbered hyperlink (for example, <a href='URL'>[1]</a>). "
        "Each citation must correspond to one of the following fields: Title, Authors, DOI, Abstract, or Published. "
        "If a citation is not already a hyperlink, manually hyperlink it to an appropriate source (for example, link 'Title' to a Google Scholar search for the title).\n\n"
        "Paper Details:\n" + let_context + "\n\n"
        "Question: " + question
    )
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": "You are an expert research assistant. Answer the research question using the provided paper details as context. For every sentence in your answer, include an inline citation formatted as a numbered hyperlink (e.g. <a href='URL'>[1]</a>, <a href='URL'>[2]</a>, etc.). Each citation must correspond to one of the following fields: Title, Authors, DOI, Abstract, or Published. If a citation is not already a hyperlink, manually hyperlink it to an appropriate source (for example, link 'Title' to a Google Scholar search for the title)."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        final_response = completion.choices[0].message.content
        citations = re.findall(r'<a\s+href="([^"]+)">\[(\d+)\]</a>', final_response)
    except Exception as e:
        final_response = f"Error retrieving response from OpenAI: {str(e)}"
        citations = []
    return jsonify({
        "response": final_response,
        "citations": citations,
        "paper_details": paper_details
    })

@app.route('/proxy')
def proxy():
    import requests
    url = request.args.get('url')
    if not url:
        return "No URL provided", 400
    try:
        resp = requests.get(url, timeout=10)
        headers = {'Content-Type': resp.headers.get('Content-Type', 'application/pdf')}
        return resp.content, resp.status_code, headers
    except Exception as e:
        return f"Error fetching URL: {e}", 500

@app.route('/latest_papers', methods=['GET'])
def latest_papers():
    """
    Scrape the arXiv API for papers released today, filter by the user's followed authors,
    and return these as the latest research. (For simplicity, only papers whose published
    date matches today's date are returned.)
    """
    today = datetime.utcnow().date()
    
    # Check if we already cached today's papers.
    if latest_papers_cache["date"] == today:
        return jsonify({"papers": latest_papers_cache["papers"]})
    
    query = "cat:cs.CL"  # Example category; adjust as needed.
    try:
        search = arxiv.Search(
            query=query,
            max_results=20,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        papers = []
        for result in search.results():
            # Use the paper's published date.
            if result.published and result.published.date() == today:
                # Filter by user's followed authors (if any)
                if followed_authors := request.args.get("followedAuthors", None):
                    # If provided as a comma-separated list, check if any is in the paper's authors.
                    authors_list = [a.strip() for a in followed_authors.split(",")]
                    if not any(author in result.title for author in authors_list):
                        continue
                papers.append({"title": result.title, "url": result.pdf_url or result.links[0].href})
        # Cache today's papers.
        latest_papers_cache["date"] = today
        latest_papers_cache["papers"] = papers
        return jsonify({"papers": papers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """
    Receives saved papers and followed authors (only the most recent two of each),
    then queries the Semantic Scholar API for similar items.
    Filters out items that the user has already read or followed.
    Additionally, occasionally recommend items based on the user's reading habits.
    """
    data = request.get_json()
    saved = data.get("savedPapers", [])
    authors_followed = data.get("followedAuthors", [])
    rec_papers = {}
    rec_authors = {}
    for paper in saved:
        query = paper
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=5&fields=title,url,year"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                results = response.json().get("data", [])
                for item in results:
                    title_rec = item.get("title")
                    url_rec = item.get("url")
                    year = item.get("year")
                    if title_rec and url_rec and title_rec not in saved:
                        if title_rec in rec_papers:
                            if rec_papers[title_rec].get("year", 0) < (year or 0):
                                rec_papers[title_rec] = {"url": url_rec, "year": year}
                        else:
                            rec_papers[title_rec] = {"url": url_rec, "year": year}
        except Exception as e:
            print("Error in recommendation for paper:", paper, e)
    for author in authors_followed:
        query = author
        url = f"https://api.semanticscholar.org/graph/v1/author/search?query={query}&limit=5&fields=name,url"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                results = response.json().get("data", [])
                for item in results:
                    name_rec = item.get("name")
                    url_rec = item.get("url")
                    if name_rec and url_rec and name_rec not in authors_followed:
                        rec_authors[name_rec] = url_rec
        except Exception as e:
            print("Error in recommendation for author:", author, e)
    if saved:
        random_saved = random.choice(saved)
        url_extra = f"https://api.semanticscholar.org/graph/v1/paper/search?query={random_saved}&limit=1&fields=title,url,year"
        try:
            response = requests.get(url_extra, timeout=5)
            if response.status_code == 200:
                results = response.json().get("data", [])
                if results:
                    item = results[0]
                    title_rec = item.get("title")
                    url_rec = item.get("url")
                    year = item.get("year")
                    if title_rec and url_rec and title_rec not in saved and title_rec not in rec_papers:
                        rec_papers[title_rec] = {"url": url_rec, "year": year}
        except Exception as e:
            print("Error in extra recommendation for reading habits:", e)
    sorted_papers = sorted(rec_papers.items(), key=lambda x: x[1].get("year", 0), reverse=True)
    recommendations = {
        "papers": [{"title": k, "url": v["url"]} for k, v in sorted_papers],
        "authors": [{"name": k, "url": v} for k, v in rec_authors.items()]
    }
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
