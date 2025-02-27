# app.py
import numpy as np
import requests
import io
import pandas as pd
import datamapplot
import urllib.parse
import xml.etree.ElementTree as ET
from flask import Flask, send_from_directory, request, jsonify

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
    f"{base_url}/raw/interactive/examples/arxiv_ml_hover_data.npy"
)
arxiv_hover_data = np.load(io.BytesIO(hover_data_file.content), allow_pickle=True)

# Build extra point data for alternative search (vertical search via topics)
topics_per_point = [
    ", ".join([label for label in labels if label != "Unlabelled"])
    for labels in zip(arxivml_label_layers[0], arxivml_label_layers[2], arxivml_label_layers[4])
]
topics_dataframe = pd.DataFrame({"topics": topics_per_point})

# ------------------------------
# Custom HTML, CSS, and JavaScript
# ------------------------------
custom_html = """
<!-- Include Google Fonts for Cinzel and Roboto -->
<link href="https://fonts.googleapis.com/css2?family=Cinzel&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">
<!-- Include Font Awesome for icons -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<div id="details-panel">
  <div id="draggable-header">DeepSeek Panel</div>
  <div id="node-details-section">
    <h2>Paper Details</h2>
    <div id="node-details">Click on a node to see paper details.</div>
  </div>
  <div id="chat-section">
    <h2>Chat Interface</h2>
    <div id="chat-container" style="display: flex; flex-direction: column; height: 400px;">
      <div id="chat-conversation" style="flex:1; overflow-y:auto; border:1px solid #ccc; padding:5px;"></div>
      <div id="chat-controls" style="margin-top:10px;">
        <textarea id="chat-input" placeholder="Ask about this paper..." style="width:100%; height:80px;"></textarea>
        <button id="chat-submit" style="width:100%; padding:8px; margin-top:4px;">Ask Question</button>
        <div id="action-buttons" style="margin-top:10px; display: flex; justify-content: space-between;">
          <button id="notifications-button" class="action-button"><i class="fa fa-bell"></i> Notifications</button>
          <button id="saved-papers-button" class="action-button">Saved Papers</button>
          <button id="authors-button" class="action-button">Followed Authors</button>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
  // Global arrays to hold saved papers, followed authors, and notifications.
  var savedPapers = [];
  var followedAuthors = [];
  var notifications = [];
  // currentPaper holds the full paper details (including the abstract).
  var currentPaper = null;
  // Track previous notification count to detect new recommendations.
  var prevNotificationCount = 0;
  
  // Replace with your actual OpenAI API key.
  const openai_api_key = "YOUR_OPENAI_API_KEY";
  
  // --- Citation Formatter ---
  function formatReply(reply) {
    return reply.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, '<a href="$2" target="_blank">$1</a>');
  }
  
  // Open the paper link in a new tab.
  function openPaper(link) {
    window.open(link, '_blank');
  }
  
  // Open a Google Scholar search for the author.
  function openScholar(author) {
    window.open('https://scholar.google.com/scholar?q=' + encodeURIComponent(author), '_blank');
  }
  
  // Toggle follow status for an individual author.
  function toggleFollowAuthor(author, event) {
    event.stopPropagation();
    var idx = followedAuthors.indexOf(author);
    if (idx === -1) {
      followedAuthors.push(author);
      alert(author + " followed!");
    } else {
      followedAuthors.splice(idx, 1);
      alert(author + " unfollowed!");
    }
    if (currentPaper) {
       fetchArxivDetails(currentPaper.title);
    }
  }
  
  // Render authors with clickable spans and follow/unfollow buttons.
  function renderAuthors(authorsString) {
    let authors = authorsString.split(',').map(a => a.trim());
    let html = "";
    authors.forEach(function(author) {
      let isFollowed = followedAuthors.includes(author);
      let buttonLabel = isFollowed ? "Unfollow" : "Follow";
      html += `<span class="clickable-author" onclick="openScholar('${author}')">${author}</span>
               <button onclick="toggleFollowAuthor('${author}', event)">${buttonLabel}</button> `;
    });
    return html;
  }
  
  // Fetch paper details from the Flask endpoint using the paper title.
  function fetchArxivDetails(hover_text) {
    console.log("Fetching details for:", hover_text);
    fetch('/get_arxiv_details?title=' + encodeURIComponent(hover_text))
      .then(response => response.json())
      .then(data => {
         if (data.error) {
             document.getElementById('node-details').innerHTML = 'Error: ' + data.error;
             return;
         }
         currentPaper = data;
         let html = `<h3><span class="clickable-paper" onclick="openPaper('${data.link}')">${data.title}</span></h3>
                     <p><strong>Abstract:</strong> ${data.abstract}</p>
                     <p><strong>Authors:</strong> ${renderAuthors(data.authors)}</p>
                     <p><strong>Published:</strong> ${data.published}</p>
                     <button onclick="saveCurrentPaper()">Save Paper</button>`;
         document.getElementById('node-details').innerHTML = html;
      })
      .catch(err => {
         console.error(err);
         document.getElementById('node-details').innerHTML = 'Error fetching details.';
      });
  }
  
  // Save the current paper if not already saved.
  function saveCurrentPaper() {
    if (currentPaper) {
      if (!savedPapers.some(p => p.title === currentPaper.title)) {
        savedPapers.push(currentPaper);
        alert("Paper saved!");
      } else {
        alert("Paper already saved.");
      }
    }
  }
  
  // Remove a saved paper.
  function removeSavedPaper(index) {
    savedPapers.splice(index, 1);
    alert("Paper removed.");
    displaySavedPapers();
  }
  
  // Display saved papers.
  function displaySavedPapers() {
    let html = "<h3>Saved Papers</h3>";
    if (savedPapers.length === 0) {
       html += "<p>No saved papers.</p>";
    } else {
       savedPapers.forEach(function(paper, idx) {
         html += `<div>
                    <span class="clickable-paper" onclick="openPaper('${paper.link}')">${paper.title}</span>
                    <button onclick="removeSavedPaper(${idx})">Remove</button>
                  </div>`;
       });
    }
    document.getElementById("node-details").innerHTML = html;
  }
  
  // Remove a followed author.
  function removeFollowedAuthor(index) {
    followedAuthors.splice(index, 1);
    alert("Author removed.");
    displayFollowedAuthors();
  }
  
  // Display followed authors.
  function displayFollowedAuthors() {
    let html = "<h3>Followed Authors</h3>";
    if (followedAuthors.length === 0) {
       html += "<p>No followed authors.</p>";
    } else {
       followedAuthors.forEach(function(author, idx) {
         html += `<div>
                    <span class="clickable-author" onclick="openScholar('${author}')">${author}</span>
                    <button onclick="removeFollowedAuthor(${idx})">Remove</button>
                  </div>`;
       });
    }
    document.getElementById("node-details").innerHTML = html;
  }
  
  // Fetch similar paper recommendations based on the current paper.
  function fetchSimilarPapers() {
    if (currentPaper && currentPaper.title) {
       fetch('/get_similar_papers?title=' + encodeURIComponent(currentPaper.title))
       .then(response => response.json())
       .then(data => {
          if(data.error) {
             console.error(data.error);
             return;
          }
          // If there are more recommendations than before, animate the notifications button.
          if (data.recommendations.length > prevNotificationCount) {
              animateNotificationsButton();
          }
          notifications = data.recommendations;
          prevNotificationCount = notifications.length;
       })
       .catch(err => console.error(err));
    }
  }
  
  // Function to add a "pop" animation to the notifications button.
  function animateNotificationsButton() {
    var button = document.getElementById("notifications-button");
    button.classList.add("pop-animation");
    button.addEventListener("animationend", function() {
       button.classList.remove("pop-animation");
    }, {once: true});
  }
  
  // Periodically fetch similar papers every 10 seconds.
  setInterval(fetchSimilarPapers, 10000);
  
  // Display notifications (similar paper recommendations).
  function displayNotifications() {
    let html = "<h3>Similar Paper Recommendations</h3>";
    if (notifications.length === 0) {
      html += "<p>No recommendations available.</p>";
    } else {
      notifications.forEach(function(rec) {
        html += `<div>
                    <span class="clickable-paper" onclick="openPaper('${rec.link}')">${rec.title}</span>
                 </div>`;
      });
    }
    document.getElementById("node-details").innerHTML = html;
  }
  
  // Chat interface.
  document.getElementById("chat-submit").addEventListener("click", function() {
    var query = document.getElementById("chat-input").value.trim();
    if (!query) return;
    
    var chatConversation = document.getElementById("chat-conversation");
    chatConversation.innerHTML += "<p><strong>You:</strong> " + query + "</p>";
    
    var messages = [
      { role: "system", content: "For every point you make, please include an inline citation formatted as markdown, e.g. [Citation](URL), that links to the relevant research paper." }
    ];
    if (currentPaper && currentPaper.abstract) {
      messages.push({ role: "system", content: "Context (paper abstract): " + currentPaper.abstract });
    }
    messages.push({ role: "user", content: query });
    
    fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
         "Content-Type": "application/json",
         "Authorization": "Bearer " + openai_api_key
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
       let formattedReply = formatReply(reply);
       chatConversation.innerHTML += "<p><strong>DeepSeek LLM:</strong> " + formattedReply + "</p>";
       document.getElementById("chat-input").value = "";
       chatConversation.scrollTop = chatConversation.scrollHeight;
    })
    .catch(error => {
       chatConversation.innerHTML += "<p style='color:red;'><strong>Error:</strong> " + error + "</p>";
    });
  });
  
  // Draggable panel functionality.
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
  
  // Event listeners for Saved Papers, Followed Authors, and Notifications buttons.
  document.getElementById("saved-papers-button").addEventListener("click", displaySavedPapers);
  document.getElementById("authors-button").addEventListener("click", displayFollowedAuthors);
  document.getElementById("notifications-button").addEventListener("click", displayNotifications);
</script>
"""

custom_css = """
/* Resizable & Draggable right panel */
#details-panel {
  position: fixed;
  top: 50px;
  right: 0;
  width: 500px;
  height: 80%;
  background: #ffffff;
  color: #000000;
  box-shadow: -4px 0 12px rgba(0,0,0,0.6);
  padding: 10px;
  z-index: 100;
  font-family: 'Cinzel', sans-serif;
  display: flex;
  flex-direction: column;
  resize: both;
  overflow: auto;
}

/* Draggable header styling */
#draggable-header {
  background: #eee;
  padding: 10px;
  cursor: move;
  font-weight: bold;
  text-align: center;
  border-bottom: 1px solid #ddd;
  user-select: none;
}

#node-details-section {
  flex: 1;
  overflow-y: auto;
  border-bottom: 1px solid #ddd;
  padding: 10px;
  margin-bottom: 10px;
}

#chat-section {
  flex: 2;
  display: flex;
  flex-direction: column;
}

#chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
}

#chat-conversation {
  flex: 1;
  overflow-y: auto;
  border: 1px solid #ccc;
  padding: 5px;
}

#chat-controls textarea {
  resize: none;
}

#action-buttons {
  margin-top: 10px;
  display: flex;
  justify-content: space-between;
}

.action-button {
  background-color: #f0f0f0;
  border: none;
  padding: 10px;
  cursor: pointer;
  flex: 1;
  margin: 0 2px;
  box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
  font-family: inherit;
  display: flex;
  align-items: center;
  justify-content: center;
}

.action-button i {
  margin-right: 5px;
}

/* Animation for notifications button */
@keyframes pop {
  0% { transform: scale(1); }
  50% { transform: scale(1.2); }
  100% { transform: scale(1); }
}

.pop-animation {
  animation: pop 0.5s ease-in-out;
}

/* Clickable elements styling */
.clickable-paper, .clickable-author {
  cursor: pointer;
  text-decoration: none;
  color: inherit;
  transition: box-shadow 0.3s ease;
}

.clickable-paper:hover, .clickable-author:hover {
  box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
}

/* Consistent font for all elements */
button, textarea, h2, div, span {
  font-family: 'Cinzel', sans-serif;
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
    font_family="Cinzel",
    hover_text=arxiv_hover_data,
    extra_point_data=topics_dataframe,
    enable_search=True,
    search_field="topics",
    on_click="fetchArxivDetails(`{hover_text}`);",
    custom_html=custom_html,
    custom_css=custom_css,
)

plot.save("static/ArXiv_data_map_example.html")

# ------------------------------
# Flask Endpoints
# ------------------------------
@app.route('/')
def index():
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
        return jsonify({"error": "No results found"}), 404

    def get_text(elem, tag, default="Not available"):
        child = elem.find('atom:' + tag, ns)
        return child.text.strip() if child is not None and child.text else default

    paper_title = get_text(entry, "title", "No title")
    paper_summary = get_text(entry, "summary", "No summary")
    paper_published = get_text(entry, "published", "No published date")

    authors = entry.findall('atom:author', ns)
    paper_authors = ", ".join([
        author.find('atom:name', ns).text.strip() if (author.find('atom:name', ns) is not None and author.find('atom:name', ns).text) else "Unknown"
        for author in authors
    ])
    paper_link = "No link available"
    for link in entry.findall('atom:link', ns):
        if link.attrib.get('rel') == 'alternate':
            paper_link = link.attrib.get('href', "No link available")
            break

    return jsonify({
        "title": paper_title,
        "abstract": paper_summary,
        "authors": paper_authors,
        "published": paper_published,
        "link": paper_link
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

if __name__ == '__main__':
    app.run(debug=True)
