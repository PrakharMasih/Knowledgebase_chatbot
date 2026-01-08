import React, { useEffect, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";


const Home = () => {
  const [chats, setChats] = useState([]);
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const navigation = useNavigate();
  const clearChats = useCallback(() => {
    setChats([]);
  }, []);

  useEffect(() => {
    const handleClearChats = () => clearChats();
    window.addEventListener('clearChats', handleClearChats);
    return () => window.removeEventListener('clearChats', handleClearChats);
  }, [clearChats]);

  useEffect(() => {
    getConversations();
  }, []);


  async function getConversations() {
    try {
      const response = await fetch("http://localhost:8000/api/v1/chat/history", {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      });

      const data = await response.json();
      console.log(data);
      setChats(data.messages ?? []);
    } catch (error) {
      console.error("Failed to fetch conversations:", error);
    }
  }

  async function handleSubmit() {
    if (!question.trim()) return;
    setIsLoading(true);
    const url = "http://localhost:8000/api/v1/chat";
    try {
      await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: JSON.stringify({
          query: question,
        }),
      });
      await getConversations();
      setQuestion("");
    } catch (error) {
      console.error("Failed to submit question:", error);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="flex items-center justify-center w-full h-screen bg-gray-100">
      <div className="w-full h-full bg-white shadow-xl overflow-hidden">
        <div className="flex flex-col h-[100vh]">
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            <div className="bg-indigo-100 p-4 mt-10 rounded-lg">
              <p className="text-indigo-800">
                How can I assist you today?
              </p>
            </div>
            {chats.map((msg, index) => (
              <div
                key={`${msg.timestamp}-${index}`}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[75%] p-3 rounded-lg ${
                    msg.role === "user"
                      ? "bg-indigo-200 text-indigo-900"
                      : "bg-gray-200 text-gray-900"
                  }`}
                >
                  {msg.content}
                </div>
              </div>
            ))}
          </div>
          <div className="p-4 border-t">
            <div className="flex space-x-2">
              <input
                className="flex-1 px-4 py-2 border rounded-full focus:outline-none focus:ring-2 focus:ring-indigo-500"
                placeholder="Ask your question here..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
              />
              <button
                className={`px-4 py-2 bg-indigo-600 text-white rounded-full hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                onClick={handleSubmit}
                disabled={isLoading}
              >
                {isLoading ? 'Sending...' : 'Send'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;