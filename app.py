import os
from main import ask_chatbot
from flask import Flask, request, jsonify, Response
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

@app.route("/chatbot", methods=["POST"])
def chatbot():
    # Twilio WhatsApp request
    if "Body" in request.values:
        user_input = request.values.get("Body", "").strip()
        response = MessagingResponse()
        if user_input.lower() in ["exit", "quit"]:
            response.message("Goodbye!")
        else:
            try:
                bot_response = ask_chatbot(user_input)
            except Exception as e:
                bot_response = "⚠️ Sorry, an error occurred."
            response.message(bot_response)
        return Response(str(response), mimetype="application/xml")
    
if __name__ == "__main__":
    app.run(port=8080, debug=True)    
