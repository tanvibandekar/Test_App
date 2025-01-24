from flask import Flask, jsonify, request

# Create the Flask app
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return "Welcome to the Flask App!"

# Define a route with JSON response
@app.route('/api/greet', methods=['GET'])
def greet():
    name = request.args.get('name', 'Guest')
    return jsonify(message=f"Hello, {name}!")

# Define a POST route to receive data
@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.json
    if not data:
        return jsonify(error="No JSON data provided"), 400
    return jsonify(message="Data received successfully", data=data)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
