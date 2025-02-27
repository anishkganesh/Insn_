# app.py
import os
import re
import io
import requests
import numpy as np
import pandas as pd
import urllib.parse
import xml.etree.ElementTree as ET
import colorcet  # for cyclic colormap support
from flask import Flask, send_from_directory, request, jsonify
from bs4 import BeautifulSoup  # for HTML scraping
import datamapplot

app = Flask(__name__, static_folder='static')

# ------------------------------
# Data Download and Processing
# ------------------------------
base_url = "https://github.com/TutteInstitute/datamapplot"

# Download data map (2D coordinates)
data_map_file = requests.get(f"{base_url}/raw/main/examples/arxiv_ml_data_map.npy")
arxivml_data_map = np.load(io.BytesIO(data_map_file.content))

# Download five layers of cluster labels
arxivml_label_layers = []
for layer_num in range(5):
    label_file = requests.get(
        f"{base_url}/raw/interactive/examples/arxiv_ml_layer{layer_num}_cluster_labels.npy"
    )
    arxivml_label_layers.append(np.load(io.BytesIO(label_file.content), allow_pickle=True))

# Download hover data (paper titles)
hover_data_file = requests.get(
    f"{base_url}/raw/main/examples/arxiv_ml_hover_data.npy"
)
arxiv_hover_data = np.load(io.BytesIO(hover_data_file.content), allow_pickle=True)

# Build extra point data (topics for vertical search)
topics_per_point = [
    ", ".join([label for label in labels if label != "Unlabelled"])
    for labels in zip(arxivml_label_layers[0], arxivml_label_layers[2], arxivml_label_layers[4])
]
topics_dataframe = pd.DataFrame({"topics": topics_per_point})

# ------------------------------
# Helper Functions
# ------------------------------
def get_text(elem, tag, default="Not available", ns={'atom': 'http://www.w3.org/2005/Atom'}):
    child = elem.find('atom:' + tag, ns)
    if child is not None and child.text:
        return " ".join(child.text.strip().split())
    return default

def scrape_arxiv_details(url, max_attempts=1):
    attempts = 0
    details = {"topic": "Not available", "citation_count": "Not available"}
    while attempts < max_attempts:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, 'html.parser')
                subject_elem = soup.find("span", class_="primary-subject")
                if subject_elem:
                    details["topic"] = subject_elem.text.strip()
                citation_elem = soup.find(string=lambda text: text and "Citations:" in text)
                if citation_elem:
                    match = re.search(r'Citations:\s*(\d+)', citation_elem)
                    if match:
                        details["citation_count"] = match.group(1)
                return details
        except Exception:
            pass
        attempts += 1
    return details

# ------------------------------
# Custom HTML, CSS, and JavaScript
# ------------------------------
custom_html = """
<!DOCTYPE html>
<html>
<head>
  <title>Insn</title>
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='0.9em' font-size='90'%3EI%3C/text%3E%3C/svg%3E">
  <link href="https://fonts.googleapis.com/css2?family=Cinzel&family=Roboto&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body>
<!-- Top Notification Box -->
<div id="topNotification"></div>

<div id="details-panel">
  <div id="draggable-header">Deep Research</div>
  <div id="node-details-section">
    <h2><i class="fa fa-file-text"></i> Paper Details</h2>
    <div id="node-details">Click on a node to see paper details.</div>
  </div>
  <div id="chat-section">
    <h2><i class="fa fa-comments"></i> Deep Research Chat</h2>
    <div id="chat-container">
      <div id="chat-conversation"></div>
      <div id="chat-controls">
        <textarea id="chat-input" placeholder="Give me basic research about..."></textarea>
        <button id="chat-submit"><i class="fa fa-paper-plane"></i> Send</button>
        <div id="action-buttons">
          <button id="notifications-button" class="action-button"><i class="fa fa-bell"></i> Alerts</button>
          <button id="saved-papers-button" class="action-button save-button"><i class="fa fa-star"></i> Favorites</button>
          <button id="authors-button" class="action-button follow-button"><i class="fa fa-user"></i> Authors</button>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
  // Note: The OpenAI API key is now stored on the server as an environment variable.
  // The client no longer has direct access to it.
  
  // Persistent cache using localStorage.
  function loadLocalCache() {
    let favs = localStorage.getItem("savedPapers");
    if (favs) { savedPapers = JSON.parse(favs); }
    let auths = localStorage.getItem("followedAuthors");
    if (auths) { followedAuthors = JSON.parse(auths); }
  }
  function updateLocalCache() {
    localStorage.setItem("savedPapers", JSON.stringify(savedPapers));
    localStorage.setItem("followedAuthors", JSON.stringify(followedAuthors));
  }
  loadLocalCache();

  // Convert markdown citations to clickable hyperlinks.
  function formatLLMReply(reply) {
    reply = reply.replace(/\*\*/g, '');
    return reply.replace(/\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g, '<a href="$2" target="_blank">[$1]</a>');
  }

  // Notification box.
  function showNotification(message) {
    const notif = document.getElementById("topNotification");
    notif.innerHTML = message;
    notif.style.display = "block";
    setTimeout(() => { notif.style.display = "none"; }, 3000);
  }

  // Global state.
  var savedPapers = [];
  var followedAuthors = [];
  var notifications = [];
  var currentPaper = null;
  var prevNotificationCount = 0;
  
  function openPaper(link) {
    window.open(link, '_blank');
  }
  function openScholar(author) {
    window.open('https://scholar.google.com/scholar?q=' + encodeURIComponent(author), '_blank');
  }
  
  // Render authors horizontally.
  function renderAuthors(authorsString) {
    let authors = authorsString.split(',').map(a => a.trim());
    let html = "";
    authors.forEach(function(author) {
      let isFollowed = followedAuthors.includes(author);
      let followIcon = isFollowed ? '<i class="fa fa-user-minus"></i>' : '<i class="fa fa-user-plus"></i>';
      html += `<div class="author-item">
                 <span class="clickable-author" onclick="openScholar('${author}')">${author}</span>
                 <button onclick="toggleFollowAuthor('${author}', event)" class="follow-button" style="padding: 4px;">${followIcon}</button>
               </div>`;
    });
    return `<div class="author-container">${html}</div>`;
  }
  
  // Fetch paper details from the backend.
  function fetchArxivDetails(hover_text) {
    fetch('/get_arxiv_details?title=' + encodeURIComponent(hover_text))
      .then(response => response.json())
      .then(data => {
         currentPaper = data;
         let linkToUse = (data.abstract && data.abstract !== "No summary" && data.abstract !== "Not available")
                         ? data.link
                         : "https://www.google.com/search?q=" + encodeURIComponent(data.title + " research paper");
         let abstractHTML = (data.abstract && data.abstract !== "No summary" && data.abstract !== "Not available")
                            ? `<p><strong>Abstract:</strong> ${data.abstract}</p>`
                            : `<p><strong>Abstract:</strong> Not available</p>`;
         let html = `<h3>
                       <span class="clickable-paper" onclick="openPaper('${linkToUse}')">${data.title}</span>
                       <button class="vertical-search" onclick="verticalSearchBasic()">Basic Research</button>
                     </h3>
                     ${abstractHTML}
                     <p><strong>Authors:</strong> ${renderAuthors(data.authors)}</p>
                     <p><strong>Published:</strong> ${data.published}</p>
                     <p><strong>Topic:</strong> ${data.topic}</p>
                     <p><strong>Citations:</strong> ${data.citation_count}</p>
                     <button onclick="saveCurrentPaper()" class="save-button">Save Paper</button>`;
         document.getElementById('node-details').innerHTML = html;
         fetchSimilarPapersForPaper(data.title);
      })
      .catch(err => {
         document.getElementById('node-details').innerHTML = 'Error fetching details.';
      });
  }
  
  // Autofill chat prompt and submit.
  function verticalSearchBasic() {
    if (currentPaper && currentPaper.title) {
      let prompt = "Give me basic research about " + currentPaper.title + ". " +
                   "Paper details: Title: " + currentPaper.title +
                   "; Authors: " + currentPaper.authors +
                   "; Published: " + currentPaper.published +
                   "; Topic: " + currentPaper.topic +
                   "; Citations: " + currentPaper.citation_count +
                   ". Please cite all your resources.";
      document.getElementById("chat-input").value = prompt;
      document.getElementById("chat-submit").click();
    }
  }
  
  function saveCurrentPaper() {
    if (currentPaper) {
      if (!savedPapers.some(p => p.title === currentPaper.title)) {
        savedPapers.push(currentPaper);
        updateLocalCache();
        showNotification("Paper saved!");
      } else {
        showNotification("Paper already saved.");
      }
    }
  }
  
  function removeSavedPaper(index) {
    savedPapers.splice(index, 1);
    updateLocalCache();
    showNotification("Paper removed.");
    displaySavedPapers();
  }
  
  function displaySavedPapers() {
    let html = "<h3>Favorites</h3>";
    if (savedPapers.length === 0) {
       html += "<p>No saved papers.</p>";
    } else {
       savedPapers.forEach(function(paper, idx) {
         html += `<div>
                    <span class="clickable-paper" onclick="openPaper('${paper.link}')">${paper.title}</span>
                    <button onclick="removeSavedPaper(${idx})" class="save-button">Remove</button>
                  </div>`;
       });
    }
    document.getElementById("node-details").innerHTML = html;
  }
  
  function toggleFollowAuthor(author, event) {
    event.stopPropagation();
    var idx = followedAuthors.indexOf(author);
    if (idx === -1) {
      followedAuthors.push(author);
      updateLocalCache();
      showNotification(author + " followed!");
    } else {
      followedAuthors.splice(idx, 1);
      updateLocalCache();
      showNotification(author + " unfollowed!");
    }
    if (currentPaper) {
       fetchArxivDetails(currentPaper.title);
    }
  }
  
  function removeFollowedAuthor(index) {
    followedAuthors.splice(index, 1);
    updateLocalCache();
    showNotification("Author removed.");
    displayFollowedAuthors();
  }
  
  function displayFollowedAuthors() {
    let html = "<h3>Authors</h3>";
    if (followedAuthors.length === 0) {
       html += "<p>No followed authors.</p>";
    } else {
       followedAuthors.forEach(function(author, idx) {
         html += `<div>
                    <span class="clickable-author" onclick="openScholar('${author}')">${author}</span>
                    <button onclick="removeFollowedAuthor(${idx})" class="follow-button">Remove</button>
                  </div>`;
       });
    }
    document.getElementById("node-details").innerHTML = html;
  }
  
  // Fetch similar papers.
  function fetchSimilarPapersForPaper(paperTitle) {
    fetch('/get_similar_papers?title=' + encodeURIComponent(paperTitle))
       .then(response => response.json())
       .then(data => {
           if (data.recommendations.length > prevNotificationCount) {
              showNotification("New similar papers available!");
           }
           notifications = data.recommendations;
           prevNotificationCount = notifications.length;
       })
       .catch(err => { console.error(err); });
  }
  
  // Chat functionality: call the backend proxy endpoint (/api/chat).
  document.getElementById("chat-submit").addEventListener("click", function() {
    var inputElem = document.getElementById("chat-input");
    var query = inputElem.value.trim();
    var context = "";
    if (currentPaper) {
      context = "Paper details: Title: " + currentPaper.title +
                "; Abstract: " + currentPaper.abstract +
                "; Authors: " + currentPaper.authors +
                "; Published: " + currentPaper.published +
                "; Topic: " + currentPaper.topic +
                "; Citations: " + currentPaper.citation_count;
    }
    if (!query && context) {
      query = "Give me basic research about " + currentPaper.title + ". Please cite all your resources.";
    }
    if (!query) return;
    
    var chatConversation = document.getElementById("chat-conversation");
    chatConversation.innerHTML += "<p><strong>You:</strong> " + query + "</p>";
    
    var messages = [
      { role: "system", content: "Context: " + context + " For every point you make, please include inline citations formatted as hyperlinked text (e.g. [Citation](URL)) for all your claims." },
      { role: "user", content: query }
    ];
    
    showNotification("Deep Research Bot is thinking...");
    
    fetch("/api/chat", {
      method: "POST",
      headers: {
         "Content-Type": "application/json"
      },
      body: JSON.stringify({
         model: "gpt-3.5-turbo",
         messages: messages,
         temperature: 0.7
      })
    })
    .then(response => response.json())
    .then(data => {
       let reply = data.choices[0].message.content;
       let formattedReply = formatLLMReply(reply);
       chatConversation.innerHTML += "<p><strong>Deep Research Bot:</strong> " + formattedReply + "</p>";
       inputElem.value = "";
       chatConversation.scrollTop = chatConversation.scrollHeight;
    })
    .catch(error => {
       chatConversation.innerHTML += "<p style='color:red;'><strong>Error:</strong> " + error + "</p>";
    });
  });

  // Submit chat on Enter key (without Shift)
  document.getElementById("chat-input").addEventListener("keydown", function(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      document.getElementById("chat-submit").click();
    }
  });
  
  // Make details panel draggable.
  function makeDraggable(el) {
    var pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
    var header = document.getElementById("draggable-header");
    if (header) {
      header.onmousedown = dragMouseDown;
    } else {
      el.onmousedown = dragMouseDown;
    }
    function dragMouseDown(e) {
      e = e || window.event;
      e.preventDefault();
      pos3 = e.clientX;
      pos4 = e.clientY;
      document.onmouseup = closeDragElement;
      document.onmousemove = elementDrag;
    }
    function elementDrag(e) {
      e = e || window.event;
      e.preventDefault();
      pos1 = pos3 - e.clientX;
      pos2 = pos4 - e.clientY;
      pos3 = e.clientX;
      pos4 = e.clientY;
      el.style.top = (el.offsetTop - pos2) + "px";
      el.style.left = (el.offsetLeft - pos1) + "px";
    }
    function closeDragElement() {
      document.onmouseup = null;
      document.onmousemove = null;
    }
  }
  makeDraggable(document.getElementById("details-panel"));
  
  document.getElementById("saved-papers-button").addEventListener("click", displaySavedPapers);
  document.getElementById("authors-button").addEventListener("click", displayFollowedAuthors);
  document.getElementById("notifications-button").addEventListener("click", function(){
      let html = "<h3>Alerts</h3>";
      if (notifications.length === 0) {
          html += "<p>No alerts available.</p>";
      } else {
          notifications.forEach(function(rec) {
              html += `<div class="similar-paper-item"><span class="clickable-paper" onclick="openPaper('${rec.link}')">${rec.title}</span></div>`;
          });
      }
      document.getElementById("node-details").innerHTML = html;
  });
</script>
</body>
</html>
"""

custom_css = """
/* Global modern styles */
body {
  margin: 0;
  font-family: 'Roboto', sans-serif;
  background-color: #f9f9f9;
  color: #333;
}

/* Top Notification Box */
#topNotification {
  position: fixed;
  top: 8px;
  left: 50%;
  transform: translateX(-50%);
  background: linear-gradient(90deg, #2980b9, #3498db);
  color: #fff;
  padding: 12px 20px;
  font-size: 16px;
  border-radius: 6px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.2);
  z-index: 1000;
  display: none;
}

/* Details Panel */
#details-panel {
  position: fixed;
  top: 50px;
  right: 20px;
  width: 520px;
  height: 80%;
  background: linear-gradient(135deg, #ffffff, #f2f2f2);
  color: #333;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  border-radius: 8px;
  padding: 15px;
  z-index: 100;
  display: flex;
  flex-direction: column;
  resize: both;
  overflow: auto;
}

/* Draggable header */
#draggable-header {
  background: linear-gradient(135deg, #2980b9, #3498db);
  padding: 12px;
  cursor: move;
  font-weight: bold;
  text-align: center;
  color: #fff;
  border-radius: 6px 6px 0 0;
  user-select: none;
}

/* Paper Details Section */
#node-details-section {
  flex: 1.5;
  overflow-y: auto;
  border-bottom: 1px solid #ddd;
  margin-bottom: 10px;
}

/* Chat Section */
#chat-section {
  flex: 1;
  display: flex;
  flex-direction: column;
}
#chat-container {
  display: flex;
  flex-direction: column;
  height: 100%;
}
#chat-conversation {
  height: 200px;
  overflow-y: auto;
  border: 1px solid #ccc;
  padding: 8px;
  background: #fff;
  border-radius: 4px;
  margin-bottom: 8px;
}
#chat-controls textarea {
  font-family: 'Roboto', sans-serif;
  border: 1px solid #ccc;
  border-radius: 4px;
  padding: 8px;
  width: 100%;
  box-sizing: border-box;
  resize: none;
}
#chat-controls button {
  width: 100%;
  padding: 10px;
  margin-top: 8px;
  border: none;
  background-color: #2980b9;
  color: #fff;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}
#chat-controls button:hover {
  background-color: #2471a3;
}

/* Action Buttons Row: add padding/gap between buttons */
#action-buttons {
  margin-top: 10px;
  display: flex;
  gap: 10px;
  justify-content: space-between;
}

/* Consistent Blue Gradient for Action, Follow, and Save Buttons */
.action-button, .follow-button, .save-button {
  background: linear-gradient(90deg, #2980b9, #3498db);
  border: none;
  color: #fff;
  padding: 10px;
  border-radius: 4px;
  cursor: pointer;
  transition: transform 0.2s ease, background 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}
.action-button:hover, .follow-button:hover, .save-button:hover {
  transform: scale(1.05);
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.action-button i, .follow-button i {
  margin-right: 5px;
}

/* Basic Research (Vertical Search) Button remains distinct */
.vertical-search {
  margin-left: 10px;
  background-color: #2ecc71;
  color: #fff;
  border: none;
  padding: 6px 10px;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}
.vertical-search:hover {
  background-color: #27ae60;
}

/* Clickable paper titles and authors */
.clickable-paper, .clickable-author {
  cursor: pointer;
  text-decoration: underline;
  color: #3498db;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.clickable-paper:hover, .clickable-author:hover {
  transform: scale(1.05);
  box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

/* Authors container styling */
.author-container {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.author-item {
  display: flex;
  align-items: center;
  gap: 4px;
}

/* Headings use Cinzel */
h1, h2, h3 {
  font-family: 'Cinzel', serif;
}
"""

# ------------------------------
# Create the Interactive Plot
# ------------------------------
plot = datamapplot.create_interactive_plot(
    arxivml_data_map,
    arxivml_label_layers[0],
    arxivml_label_layers[2],
    arxivml_label_layers[4],
    font_family="Roboto",
    hover_text=arxiv_hover_data,
    extra_point_data=topics_dataframe,
    enable_search=True,
    search_field="topics",
    on_click="fetchArxivDetails(`{hover_text}`);",
    custom_html=custom_html,
    custom_css=custom_css,
    cmap=colorcet.cm.colorwheel,
    cluster_boundary_polygons=True,
    cluster_boundary_line_width=8,
    background_color="#eeeeee"
)

plot.save("static/ArXiv_data_map_example.html")

# ------------------------------
# Flask Endpoints
# ------------------------------
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/app')
def app_route():
    return send_from_directory('static', 'ArXiv_data_map_example.html')

@app.route('/get_arxiv_details')
def get_arxiv_details():
    title = request.args.get('title', '')
    if not title:
        return jsonify({"error": "No title provided"}), 400

    query = 'ti:"{}"'.format(title)
    query_url = "http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(query) + "&max_results=1"
    response = requests.get(query_url)
    if response.status_code != 200:
        return jsonify({"error": "Error fetching data from arXiv"}), 500

    try:
        root = ET.fromstring(response.content)
    except Exception as e:
        return jsonify({"error": "Error parsing XML: " + str(e)}), 500

    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    entry = root.find('atom:entry', ns)
    if entry is None:
        return jsonify({
            "title": title,
            "abstract": "Not available",
            "authors": "Not available",
            "published": "Not available",
            "link": "No link available",
            "topic": "Not available",
            "citation_count": "Not available"
        })

    paper_title = get_text(entry, "title", "No title", ns)
    paper_summary = get_text(entry, "summary", "No summary", ns)
    paper_published = get_text(entry, "published", "No published date", ns)
    authors = entry.findall('atom:author', ns)
    paper_authors = ", ".join([
        author.find('atom:name', ns).text.strip() if (author.find('atom:name', ns) is not None and author.find('atom:name', ns).text) else "Unknown"
        for author in authors
    ])
    candidate_links = [link.attrib.get('href', "") for link in entry.findall('atom:link', ns) if link.attrib.get('rel') == 'alternate']
    paper_link = "No link available"
    scraped_details = {"topic": "Not available", "citation_count": "Not available"}
    if candidate_links:
        for candidate in candidate_links[:4]:
            details = scrape_arxiv_details(candidate, max_attempts=1)
            if details["topic"] != "Not available" or details["citation_count"] != "Not available":
                paper_link = candidate
                scraped_details = details
                break
        if paper_link == "No link available":
            paper_link = candidate_links[0]

    return jsonify({
        "title": paper_title,
        "abstract": paper_summary,
        "authors": paper_authors,
        "published": paper_published,
        "link": paper_link,
        "topic": scraped_details["topic"],
        "citation_count": scraped_details["citation_count"]
    })

@app.route('/get_similar_papers')
def get_similar_papers():
    title = request.args.get('title', '')
    if not title:
        return jsonify({"error": "No title provided"}), 400

    words = title.split()
    keywords = " ".join(words[:3]) if len(words) >= 3 else title
    query = 'ti:"{}"'.format(keywords)
    query_url = "http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(query) + "&max_results=5"
    response = requests.get(query_url)
    if response.status_code != 200:
        return jsonify({"error": "Error fetching data from arXiv"}), 500
    try:
        root = ET.fromstring(response.content)
    except Exception as e:
        return jsonify({"error": "Error parsing XML: " + str(e)}), 500
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    entries = root.findall('atom:entry', ns)
    recommendations = []
    for entry in entries:
        paper_title = entry.find('atom:title', ns).text.strip() if entry.find('atom:title', ns) is not None else "No title"
        if paper_title.lower() == title.lower():
            continue
        paper_summary = entry.find('atom:summary', ns).text.strip() if entry.find('atom:summary', ns) is not None else "No summary"
        paper_link = "No link available"
        for link in entry.findall('atom:link', ns):
            if link.attrib.get('rel') == 'alternate':
                paper_link = link.attrib.get('href', "No link available")
                break
        recommendations.append({
            "title": paper_title,
            "abstract": paper_summary,
            "link": paper_link
        })
    return jsonify({"recommendations": recommendations})

# New backend endpoint that proxies the OpenAI API request.
@app.route('/api/chat', methods=['POST'])
def api_chat():
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if not openai_api_key:
        return jsonify({"error": "API key not configured"}), 500
    payload = request.json
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai_api_key
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(debug=True)
