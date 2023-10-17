from flask import Flask, request
from main import calculate

app = Flask(__name__)

@app.route('/sarcasm_percentage', methods=['POST'])
def my_post_req():
    if request.method == 'POST':
        text = request.form.get('text')

        return str(calculate(text)), 200

    return 'Err', 405

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3200)

