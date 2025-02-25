import numpy as np
import requests
import io
import pandas as pd
import datamapplot

# --- Download the data ---
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

# Download hover data (e.g. paper titles)
hover_data_file = requests.get(
    f"{base_url}/raw/interactive/examples/arxiv_ml_hover_data.npy"
)
arxiv_hover_data = np.load(io.BytesIO(hover_data_file.content), allow_pickle=True)

# --- Build extra point data for alternative search ---
# For each point, join the labels (skipping "Unlabelled") from three layers
topics_per_point = [
    ", ".join([label for label in labels if label != "Unlabelled"])
    for labels in zip(arxivml_label_layers[0], arxivml_label_layers[2], arxivml_label_layers[4])
]
topics_dataframe = pd.DataFrame({"topics": topics_per_point})

# --- Custom HTML and CSS for the right-side panel with DeepSeek LLM chat integration ---
custom_html = """
<!-- Include Font Awesome for icons -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<div id="details-panel">
  <div id="draggable-header">DeepSeek Panel</div>
  <div id="node-details-section">
    <h2>Node Details</h2>
    <div id="node-details">Click on a node to see details.</div>
  </div>
  <div id="chat-section">
    <h2>Chat Interface</h2>
    <div id="chat-container" style="display: flex; flex-direction: column; height: 400px;">
      <div id="chat-conversation" style="flex:1; overflow-y:auto; border:1px solid #ccc; padding:5px;"></div>
      <div id="chat-controls" style="margin-top:10px;">
        <textarea id="chat-input" placeholder="Ask about a topic or paper..." style="width:100%; height:80px;"></textarea>
        <button id="chat-submit" style="width:100%; padding:8px; margin-top:4px;">Search & Chat</button>
        <div id="action-buttons" style="margin-top:10px; display: flex; justify-content: space-between;">
          <button id="notifications-button" class="action-button"><i class="fa fa-bell"></i> Notifications</button>
          <button id="saved-papers-button" class="action-button"><i class="fa fa-save"></i> Saved Papers</button>
          <button id="authors-button" class="action-button"><i class="fa fa-user"></i> Authors</button>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
  // Replace with your actual OpenAI API key.
  const openai_api_key = "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";

  // Function to process reply text and replace links with inline citation numbers.
  function formatReply(reply) {
    let citationCount = 0;
    // Regex to match anchor tags.
    return reply.replace(/<a\\s+href="([^"]+)"[^>]*>(.*?)<\\/a>/g, function(match, url, text) {
      citationCount++;
      return `<sup><a href="${url}" target="_blank" style="text-decoration:none;">[${citationCount}]</a></sup>`;
    });
  }

  // Chat interface: when the button is clicked, send the query to the OpenAI API.
  document.getElementById("chat-submit").addEventListener("click", function() {
    var query = document.getElementById("chat-input").value.trim();
    if (!query) return;

    var chatConversation = document.getElementById("chat-conversation");
    chatConversation.innerHTML += "<p><strong>You:</strong> " + query + "</p>";

    // API call with system instruction for citations.
    fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
         "Content-Type": "application/json",
         "Authorization": "Bearer " + openai_api_key
      },
      body: JSON.stringify({
         model: "gpt-3.5-turbo",
         messages: [
           {
             role: "system", 
             content: "For every point you make, please include a citation from a research paper. Ensure that every citation is a clickable link to the research paper's page. Cite where you get every point from."
           },
           { role: "user", content: query }
         ],
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

/* Ensure links in chat are clickable */
#chat-conversation a {
  pointer-events: auto;
  color: blue;
  text-decoration: underline;
}

/* Consistent font for all elements */
button, textarea, h2, div {
  font-family: 'Cinzel', sans-serif;
}
"""

# --- Create the interactive plot ---
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
    on_click="document.getElementById('node-details').innerHTML = `{hover_text}`",
    custom_html=custom_html,
    custom_css=custom_css,
)

plot.save("static/ArXiv_data_map_example.html")