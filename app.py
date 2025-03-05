import os
import re
import json
import io
import concurrent.futures
from flask import Flask, render_template, request, jsonify
import numpy as np, requests
import datamapplot, arxiv, openai
from requests.adapters import HTTPAdapter
from difflib import SequenceMatcher
from bs4 import BeautifulSoup

app = Flask(__name__)

# Load OpenAI API key from environment variable.
openai.api_key = os.getenv("OPENAI_API_KEY")

# In-memory cache for paper details.
paper_cache = {}

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

def best_match(title, candidates, threshold=0.5):
    best_candidate, best_ratio = None, 0.0
    for candidate in candidates:
        ratio = SequenceMatcher(None, title.lower(), candidate.title.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_candidate = candidate
    return best_candidate if best_ratio >= threshold else None

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
    query_variants = [
        f'ti:"{title}"',
        f'all:"{title}"',
        f'ti:{title}',
        title  # raw title search.
    ]
    best_candidate = None
    best_ratio = 0.0
    threshold = 0.75
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(query_variants)) as executor:
        futures = {executor.submit(search_variant, query, title): query for query in query_variants}
        for future in concurrent.futures.as_completed(futures):
            candidate, ratio = future.result()
            if ratio > best_ratio:
                best_ratio = ratio
                best_candidate = candidate
    if best_candidate is not None and best_ratio >= threshold:
        return best_candidate
    return None

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
        "You are an expert research assistant. Using the paper details provided below as context, answer the following research question. "
        "For every sentence in your answer, include an inline citation in the format [Source: <Field>], where <Field> is one of: Title, Authors, DOI, Abstract, or Published. "
        "Your answer must be fully annotated with these citations.\n\n"
        "Paper Details:\n" + let_context + "\n\n"
        "Question: " + question
    )
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": "You are an expert research assistant who always includes inline citations for every sentence based on the paper details provided."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        final_response = completion.choices[0].message.content
        print("Chat response from OpenAI:", final_response)
        citations = re.findall(r'\[Source:\s*(.*?)\]', final_response)
    except Exception as e:
        final_response = f"Error retrieving response from OpenAI: {str(e)}"
        citations = []
    return jsonify({
        "response": final_response,
        "citations": citations,
        "paper_details": paper_details
    })

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """
    Receives saved papers and followed authors (only the most recent two of each),
    then queries the Semantic Scholar API for similar items.
    Filters duplicates and returns unique recommendations.
    """
    data = request.get_json()
    saved = data.get("savedPapers", [])[-2:]
    authors = data.get("followedAuthors", [])[-2:]
    rec_papers = {}
    rec_authors = {}
    for paper in saved:
        query = paper
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=2&fields=title,url"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                results = response.json().get("data", [])
                for item in results:
                    title_rec = item.get("title")
                    url_rec = item.get("url")
                    if title_rec and url_rec:
                        rec_papers[title_rec] = url_rec
        except Exception as e:
            print("Error in recommendation for paper:", paper, e)
    for author in authors:
        query = author
        url = f"https://api.semanticscholar.org/graph/v1/author/search?query={query}&limit=2&fields=name,url"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                results = response.json().get("data", [])
                for item in results:
                    name_rec = item.get("name")
                    url_rec = item.get("url")
                    if name_rec and url_rec:
                        rec_authors[name_rec] = url_rec
        except Exception as e:
            print("Error in recommendation for author:", author, e)
    recommendations = {
        "papers": [{"title": k, "url": v} for k, v in rec_papers.items()],
        "authors": [{"name": k, "url": v} for k, v in rec_authors.items()]
    }
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
