import React, { useState, useEffect, useRef } from "react";
import { MdPerson, MdAndroid } from "react-icons/md";
import "./chatbot.css";

const Chatbot = () => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [data, setData] = useState({});
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault(); // Prevent default form submission

    // Add user message to messages state
    const userMessage = { text: input, user: true };
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    // Clear input field
    setInput("");

    try {
      // Make a POST request to the server endpoint ("/")
      const response = await fetch("/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json", // Set content type to JSON
        },
        body: JSON.stringify({ message: input }), // Send user input as JSON data
      });

      // Check if the request was successful (status code 200)
      if (response.ok) {
        // Parse the JSON response
        const responseData = await response.json();

        // Update the state with the response data
        setData(responseData);

        // Add bot response to messages state
        const botMessage = { text: responseData.message, user: false };
        setMessages((prevMessages) => [...prevMessages, botMessage]);

        // Log the response data
        console.log(responseData);
      } else {
        // If there's an error, log it
        console.error("Failed to fetch data:", response.statusText);
      }
    } catch (error) {
      // If there's an error, log it
      console.error("Error fetching data:", error);
    }
  };

  return (
    <div className="chatbot-container">
      <h1 className="head">AI Chatbot</h1>
      <div className="chatbot-messages">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`message ${
              message.user ? "user-message" : "ai-message"
            }`}
          >
            <span className="message-text">{message.text}</span>
            {message.user ? (
              <span className="icon-container">
                <MdPerson className="user-icon" />
              </span>
            ) : (
              <span className="icon-container">
                <MdAndroid className="ai-icon" />
              </span>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <form className="chatbot-input-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
        />
        <button type="submit">Send</button>
      </form>
      <div>{data.message}</div> {/* Display server response */}
    </div>
  );
};

export default Chatbot;
