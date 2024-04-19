from flask import Flask, render_template, request, jsonify
from sentiment_analysis import predict

app = Flask(__name__)

@app.route("/", methods=["POST"])
def process_user_message():
    try:
        # Get the user message from the JSON request data
        data = request.get_json()
        user_message = data["message"]

        # Process the user message (you can replace this function with your chatbot logic)
        bot_response = chatbot_logic(user_message)

        # Return the response as JSON
        return jsonify({"message": bot_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def chatbot_logic(user_message):
    return predict(user_message)

if __name__ == '__main__':
    app.run(debug=True)