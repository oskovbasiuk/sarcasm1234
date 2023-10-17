from flask_cors import CORS, cross_origin
from flask import Flask, request
from main import calculate

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
@app.route('/sarcasm_percentage', methods=['POST'])
@cross_origin()
def my_post_req():
    if request.method == 'POST':
        text = request.json['text']

        return str(calculate(text)), 200

    return 'Err', 405

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3200)

