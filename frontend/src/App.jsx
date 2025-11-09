import { useState, useEffect, useRef } from "react";
import "./App.css"; // Import our new styles
import ReactMarkdown from "react-markdown"; // For rendering bullet points, etc.

// --- Helper: Generate a simple random session ID ---
function generateSessionId() {
  return "session_" + Math.random().toString(36).substring(2, 15);
}

function App() {
  const [isOpen, setIsOpen] = useState(false); // Controls the popup
  const [sessionId, setSessionId] = useState("");
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState([
    {
      sender: "ai",
      text: "Hi! I'm the Genilia assistant. How can I help you today?",
    },
  ]);

  const messagesEndRef = useRef(null); // Ref to auto-scroll

  // Generate a session ID once on page load
  useEffect(() => {
    setSessionId(generateSessionId());
  }, []);

  // Auto-scroll to the bottom when new messages appear
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    const userMessage = { sender: "user", text: inputValue };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInputValue("");
    setIsLoading(true);

    // --- THIS IS THE API CALL TO OUR MCP ---
    try {
      const response = await fetch("http://127.0.0.1:8002/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          input: inputValue,
          session_id: sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();

      // Get the answer from either the RAG or Action agent
      const aiText = data.answer || data.output || "Sorry, something went wrong.";

      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "ai", text: aiText },
      ]);
    } catch (error) {
      console.error("Error fetching from MCP:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          sender: "ai",
          text: "I'm having trouble connecting to my brain. Please try again later.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
    // --- END OF API CALL ---
  };

  return (
    <div className="App">
      {/* --- The Popup Chat Window --- */}
      <div className={`chat-popup ${isOpen ? "open" : ""}`}>
        <div className="chat-header">Genilia Support Agent</div>
        <div className="chat-messages">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
              {/* Use ReactMarkdown to render formatted text */}
              <ReactMarkdown>{msg.text}</ReactMarkdown>
            </div>
          ))}
          {/* This empty div is our scroll target */}
          <div ref={messagesEndRef} />
        </div>
        <form className="chat-input" onSubmit={handleSubmit}>
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder={isLoading ? "Waiting for response..." : "Ask a question..."}
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading}>
            ➤
          </button>
        </form>
      </div>

      {/* --- The Toggle Button --- */}
      <button className="chat-button" onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? "✕" : "💬"}
      </button>
    </div>
  );
}

export default App;