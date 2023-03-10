from flask import Flask, request
from model import getResponse
import webbrowser

app = Flask(__name__)



@app.route("/translate" , methods=["POST"])
def translate_text():
    input_text = request.json.get("input_text")
    response = getResponse(input_text)
    return {"response": response}

@app.route("/")
def index():
    return open("index.html").read()

if __name__ == '__main__':
    app.run() # http://127.0.0.1:5000/
    #while True:
     #  print(getResponse(input("Enter text: ")))
        
        
    

