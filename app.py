from flask import Flask, send_from_directory

app = Flask(__name__, static_folder="static")

@app.route('/')
def index():
    # Serve the interactive plot HTML from the static folder
    return send_from_directory(app.static_folder, "ArXiv_data_map_example.html")

if __name__ == '__main__':
    app.run(debug=True)