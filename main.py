import os
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Optional
import requests
import base64
import io
import json
import logging
from logging.handlers import RotatingFileHandler

# LangChain imports
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage

# Google Gemini imports
import google.generativeai as genai
from PIL import Image

# LiveKit imports
from livekit import api

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('agent.log', maxBytes=10485760, backupCount=5),  # 10MB per file, 5 backups
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multimodal AI Agent with LangChain & Gemini")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistent chat history storage
HISTORY_FILE = "chat_history.json"

def load_chat_history():
    """Load chat history from file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_chat_history():
    """Save chat history to file"""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving chat history: {e}")

# Load existing chat history on startup
chat_history: List[dict] = load_chat_history()
logger.info(f"Loaded {len(chat_history)} chat history items from file")

def get_recent_conversation_history(n: int = 5) -> List:
    """
    Get the last N conversations for short-term memory.
    Returns LangChain message format for the agent.
    """
    if not chat_history:
        return []
    
    # Get last N conversations
    recent = chat_history[-n:] if len(chat_history) > n else chat_history
    
    # Convert to LangChain message format
    messages = []
    for item in recent:
        messages.append(HumanMessage(content=item["query"]))
        messages.append(AIMessage(content=item["answer"]))
    
    logger.debug(f"Retrieved {len(messages)//2} recent conversations for context")
    return messages

# Request model
class QueryRequest(BaseModel):
    query: str

# Response model
class QueryResponse(BaseModel):
    reasoning: str
    answer: str

# Chat history item model
class ChatHistoryItem(BaseModel):
    id: int
    query: str
    reasoning: str
    answer: str
    timestamp: str

# Get API keys
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")

# Initialize Gemini
gemini_client = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_client = genai.GenerativeModel('gemini-2.0-flash-exp')
    logger.info("Gemini client initialized successfully")
else:
    logger.warning("GEMINI_API_KEY not found - AI features will be limited")

# Initialize LangChain LLM with Gemini
llm = None
if GEMINI_API_KEY:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=GEMINI_API_KEY,
        temperature=0.7,
        convert_system_message_to_human=True
    )
    logger.info("LangChain LLM (Gemini) initialized successfully")
else:
    logger.warning("LLM not initialized - GEMINI_API_KEY required")

# Tool functions
def get_weather(city: str) -> str:
    """Get current weather for a city using OpenWeatherMap API."""
    logger.info(f"Tool called: get_weather(city='{city}')")
    
    weather_api_key = os.getenv("OPENWEATHER_API_KEY")
    if not weather_api_key or weather_api_key.strip() == "":
        logger.error("OpenWeather API key not configured - cannot fetch weather")
        raise Exception("Weather API is not configured. Please add OPENWEATHER_API_KEY to your .env file to use weather features.")
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&units=metric"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        temp = data["main"]["temp"]
        description = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        
        result = f"Weather in {city}: {temp}°C, {description}, Humidity: {humidity}%"
        logger.info(f"Weather data retrieved successfully for {city}")
        return result
    except requests.RequestException as e:
        logger.error(f"Error fetching weather: {str(e)}")
        raise Exception(f"Unable to fetch weather data for {city}. Error: {str(e)}")
    except KeyError:
        logger.error(f"Invalid city name or API response format for {city}")
        raise Exception(f"City '{city}' not found or invalid API response.")
    except Exception as e:
        logger.error(f"Error fetching weather: {str(e)}")
        raise Exception(f"Unable to fetch weather data. Error: {str(e)}")

def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information about a topic."""
    logger.info(f"Tool called: search_wikipedia(query='{query}')")
    
    try:
        wiki = MediaWiki()
        search_results = wiki.search(query, results=3)
        
        if not search_results:
            logger.info(f"No Wikipedia results found for: {query}")
            return f"No information found on Wikipedia for '{query}'."
        
        # Get the first result
        page = wiki.page(search_results[0])
        summary = page.summary[:500]  # First 500 characters
        
        logger.info(f"Wikipedia summary retrieved successfully for: {query}")
        return f"Wikipedia: {summary}... (Source: {page.url})"
    except Exception as e:
        logger.error(f"Error searching Wikipedia: {str(e)}")
        return f"Unable to fetch information from Wikipedia. Error: {str(e)}"

def get_news(topic: str) -> str:
    """Get latest news articles about a topic."""
    logger.info(f"Tool called: get_news(topic='{topic}')")
    
    news_api_key = os.getenv("NEWS_API_KEY")
    if not news_api_key or news_api_key.strip() == "":
        logger.error("NEWS_API_KEY not configured - cannot fetch news")
        raise Exception("News API is not configured. Please add NEWS_API_KEY to your .env file to use news features.")
    
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": topic,
            "apiKey": news_api_key,
            "pageSize": 3,
            "sortBy": "publishedAt"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        articles = data.get("articles", [])
        
        if not articles:
            logger.info(f"No news articles found for: {topic}")
            return f"No recent news found for '{topic}'."
        
        news_items = []
        for article in articles[:3]:
            news_items.append(f"- {article['title']} ({article['source']['name']})")
        
        logger.info(f"Retrieved {len(news_items)} news articles for: {topic}")
        return "Latest news:\n" + "\n".join(news_items)
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        raise Exception(f"Unable to fetch news. Error: {str(e)}")

def transcribe_audio(audio_base64: str) -> str:
    """Transcribe audio to text - placeholder for LiveKit integration."""
    logger.info("Tool called: transcribe_audio")
    
    # Note: Gemini File API is having issues with audio files
    # This should be replaced with LiveKit audio streaming or Web Speech API
    logger.warning("Audio transcription via Gemini File API is currently unavailable")
    
    return "Audio transcription is not available. Please use text input or configure LiveKit for real-time audio streaming."

def analyze_image(image_base64: str, query: str = "What do you see in this image?") -> str:
    """Analyze image using Gemini's vision capabilities."""
    logger.info(f"Tool called: analyze_image(query='{query}')")
    
    if not gemini_client:
        logger.warning("Gemini API key not configured for image analysis")
        return "Gemini API key is not configured for image analysis."
    
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Generate response with Gemini Vision
        response = gemini_client.generate_content([query, image])
        
        logger.info("Image analysis completed successfully")
        return response.text
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        return f"Unable to analyze image. Error: {str(e)}"

def detect_hand_signs(image_base64: str) -> str:
    """Detect and interpret hand signs from image using Gemini Vision."""
    logger.info("Tool called: detect_hand_signs")
    
    if not gemini_client:
        logger.warning("Gemini API key not configured for hand sign detection")
        return "Gemini API key is not configured for hand sign detection."
    
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Generate response
        prompt = "Analyze this image and identify any hand signs, gestures, or sign language. Describe what the hand gesture means or represents. Be specific about the hand position, fingers, and any recognized signs."
        response = gemini_client.generate_content([prompt, image])
        
        logger.info("Hand sign detection completed successfully")
        return response.text
    except Exception as e:
        logger.error(f"Error detecting hand signs: {str(e)}")
        return f"Unable to detect hand signs. Error: {str(e)}"

def text_to_speech(text: str) -> str:
    """
    Convert text to speech using Google Text-to-Speech (gTTS).
    Returns base64 encoded audio data.
    """
    logger.info(f"Converting text to speech: {text[:50]}...")
    
    try:
        # Using Google's free TTS (gTTS)
        from gtts import gTTS
        import tempfile
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_path = fp.name
        
        # Generate speech
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(temp_path)
        
        # Read and encode audio
        with open(temp_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Clean up
        os.remove(temp_path)
        
        logger.info("Text-to-speech conversion successful")
        return audio_base64
    except ImportError:
        logger.warning("gTTS library not installed - text-to-speech unavailable")
        return ""
    except Exception as e:
        logger.error(f"Error in text-to-speech: {str(e)}")
        return ""

# Create LangChain tools (only text, audio, video - no image analysis)
tools = [
    Tool(
        name="get_weather",
        func=get_weather,
        description="Useful for getting current weather information for a city. Input should be a city name. Requires OPENWEATHER_API_KEY to be configured."
    ),
    Tool(
        name="search_wikipedia",
        func=search_wikipedia,
        description="Useful for finding factual information about people, places, things, or concepts. Input should be a search query or topic."
    ),
    Tool(
        name="get_news",
        func=get_news,
        description="Useful for getting the latest news articles about a specific topic. Input should be the topic or keyword you want news about. Requires NEWS_API_KEY to be configured."
    ),
    Tool(
        name="transcribe_audio",
        func=transcribe_audio,
        description="Useful for converting voice/audio to text via LiveKit or Gemini. Input should be base64 encoded audio data."
    )
]

# Create agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional AI assistant powered by Google Gemini with expertise in multimodal interactions.

CORE CAPABILITIES:
• Real-time video analysis and computer vision
• Voice transcription and natural language processing
• Screen share content analysis
• Multi-turn conversation with context awareness
• Information retrieval (weather, news, Wikipedia)

COMMUNICATION GUIDELINES:
• Maintain a professional, clear, and concise tone
• Provide accurate, well-structured responses
• Use proper grammar and formatting
• Avoid casual language or excessive emoji usage
• Be direct and informative without being verbose

MEMORY & CONTEXT HANDLING:
• Access conversation history before responding "I don't know"
• Recall user-provided information (name, preferences, previous discussions)
• Reference past interactions when relevant to maintain conversational continuity
• Acknowledge when information was shared in earlier exchanges

VIDEO & VISUAL ANALYSIS:
• Analyze live camera feed or screen share content when user asks questions
• Provide detailed, objective descriptions of visual content
• Identify objects, text, code, gestures, or any visible elements
• Maintain professionalism in all visual interpretations

RESPONSE PROTOCOL:
• For factual queries: Use appropriate tools (weather, news, Wikipedia)
• For visual queries: Analyze camera or screen share feed
• For errors: Communicate issues clearly and suggest solutions
• For missing information: Check conversation history first, then ask for clarification

Always prioritize accuracy, clarity, and professional communication standards."""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create agent
agent = None
agent_executor = None
if llm:
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=3,
        handle_parsing_errors=True,
        return_intermediate_steps=True  # To capture tool errors
    )

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Main endpoint to handle user queries using LangChain agent with Gemini"""
    logger.info(f"Received query: {request.query[:100]}...")  # Log first 100 chars
    
    if not request.query or not request.query.strip():
        logger.warning("Empty query received")
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if not agent_executor:
        logger.error("Agent executor not configured - missing GEMINI_API_KEY")
        raise HTTPException(
            status_code=500, 
            detail="AI agent is not configured. Please add GEMINI_API_KEY to your .env file."
        )
    
    try:
        query_input = request.query
        
        # Get recent conversation history for short-term memory
        recent_history = get_recent_conversation_history(20)  # Last 20 conversations for better memory
        
        # Use LangChain agent to process the query with conversation context
        logger.debug(f"Invoking agent with {len(recent_history)//2} previous conversations")
        result = agent_executor.invoke({
            "input": query_input,
            "chat_history": recent_history  # Pass short-term memory
        })
        
        # Extract reasoning from intermediate steps
        reasoning = "Used LangChain agent (Gemini) with tools: "
        if "intermediate_steps" in result and result["intermediate_steps"]:
            tools_used = [step[0].tool for step in result["intermediate_steps"]]
            reasoning += ", ".join(tools_used)
            logger.info(f"Tools used: {', '.join(tools_used)}")
        else:
            reasoning = "Processed query with LangChain agent (Gemini)"
            logger.info("No tools used - direct LLM response")
        
        answer = result.get("output", "No response generated")
        
        # Generate text-to-speech audio for the response
        audio_base64 = text_to_speech(answer)
        
        # Store in chat history
        history_item = {
            "id": len(chat_history) + 1,
            "query": request.query,
            "reasoning": reasoning,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "audio": audio_base64 if audio_base64 else None
        }
        chat_history.append(history_item)
        save_chat_history()  # Save to file
        logger.info(f"Response generated and saved to history (ID: {history_item['id']})")
        
        # Include audio in response
        response_data = QueryResponse(reasoning=reasoning, answer=answer)
        if audio_base64:
            # Add audio field to response
            return {
                "reasoning": reasoning,
                "answer": answer,
                "audio": audio_base64
            }
        return response_data
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Multimodal request models - MUST BE BEFORE ENDPOINTS THAT USE THEM
class AudioRequest(BaseModel):
    audio_data: str  # base64 encoded audio
    query: Optional[str] = None

class VideoFrameRequest(BaseModel):
    frame_data: str  # base64 encoded image
    query: Optional[str] = None

class ContinuousSignRequest(BaseModel):
    frames: List[str]  # List of base64 encoded images (frames)
    
class ImageRequest(BaseModel):
    image_data: str  # base64 encoded image
    query: Optional[str] = None
    mode: str = "analyze"  # analyze or hand_signs

@app.post("/ask-with-video")
async def ask_with_video(request: VideoFrameRequest):
    """Handle queries with video frame - analyze what user is showing"""
    if not gemini_client:
        raise HTTPException(status_code=500, detail="Gemini AI not configured")
    
    try:
        # Decode base64 image - handle data URL format
        frame_data = request.frame_data
        if ',' in frame_data:
            # Remove data URL prefix (e.g., "data:image/jpeg;base64,")
            frame_data = frame_data.split(',', 1)[1]
        
        # Decode base64
        image_bytes = base64.b64decode(frame_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get conversation history for context
        recent_history = get_recent_conversation_history(10)
        history_text = ""
        if recent_history:
            history_text = "\n\n**CONVERSATION CONTEXT:**\n"
            for i in range(0, len(recent_history), 2):
                if i+1 < len(recent_history):
                    user_msg = recent_history[i].content[:100]
                    ai_msg = recent_history[i+1].content[:100]
                    history_text += f"• User: {user_msg}\n• You: {ai_msg}\n"
        
        # Analyze with Gemini Vision - General video analysis
        query = request.query or "What do you see?"
        prompt = f"""You are a professional AI assistant with computer vision capabilities.
{history_text}

**USER QUERY:** {query}

**ANALYSIS REQUIREMENTS:**
Examine the provided image and deliver a clear, professional response addressing the user's specific question.

**POTENTIAL VISUAL CONTENT:**
• Objects, items, or products
• Hand gestures or body language
• Environmental context or settings
• Text, documents, or written content
• Code, technical diagrams, or interfaces
• Any visual elements requiring identification or explanation

**RESPONSE PROTOCOL:**
✓ Provide direct, factual answers without unnecessary preamble
✓ Use specific, descriptive language for visual elements
✓ Maintain professional tone throughout
✓ Reference conversation history when contextually relevant
✓ Structure complex responses with clear organization

✗ Avoid phrases like "I can see that..." or "It appears to be..."
✗ Do not over-describe; focus on answering the query
✗ Avoid casual language or excessive informality

**RESPONSE EXAMPLES:**
Query: "What am I holding?" → "You're holding a [specific object description]"
Query: "Identify this object" → "[Object name/type] - [brief relevant details]"
Query: "What gesture is this?" → "This is a [gesture name] gesture, commonly used for [meaning/context]"

Analyze the image and provide your professional assessment."""
        
        response = gemini_client.generate_content([prompt, image])
        answer = response.text.strip()
        
        logger.info(f"Video query answered: {answer[:100]}...")
        
        # Generate audio
        audio_base64 = text_to_speech(answer)
        
        # Store in history
        history_item = {
            "id": len(chat_history) + 1,
            "query": f"[Video] {query}",
            "reasoning": "Analyzed video frame with Gemini Vision for hand signs",
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "audio": audio_base64
        }
        chat_history.append(history_item)
        save_chat_history()
        
        return {
            "reasoning": "Analyzed video frame with Gemini Vision",
            "answer": answer,
            "audio": audio_base64
        }
        
    except Exception as e:
        logger.error(f"Error in video query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/ask-about-screen")
async def ask_about_screen(request: VideoFrameRequest):
    """Handle queries about screen share content - analyze what's visible on screen"""
    if not gemini_client:
        raise HTTPException(status_code=500, detail="Gemini AI not configured")
    
    try:
        logger.info(f"Screen content query received: {request.query}")
        
        # Decode the frame
        frame_data = request.frame_data
        if ',' in frame_data:
            frame_data = frame_data.split(',', 1)[1]
        
        image_bytes = base64.b64decode(frame_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get conversation history for context
        recent_history = get_recent_conversation_history(10)
        history_text = ""
        if recent_history:
            history_text = "\n\n**CONVERSATION CONTEXT:**\n"
            for i in range(0, len(recent_history), 2):
                if i+1 < len(recent_history):
                    user_msg = recent_history[i].content[:100]
                    ai_msg = recent_history[i+1].content[:100]
                    history_text += f"• User: {user_msg}\n• You: {ai_msg}\n"
        
        # Create prompt for screen content analysis
        prompt = f"""You are a professional AI assistant with screen analysis capabilities.
{history_text}

**USER QUERY:** {request.query}

**ANALYSIS OBJECTIVE:**
Examine the screen share content and provide a clear, professional response to the user's query.

**SCREEN CONTENT CATEGORIES:**
• Code Editors (VS Code, IDEs): Code review, debugging assistance, optimization suggestions
• Documents & Text: Content analysis, summarization, interpretation
• Web Applications: UI/UX analysis, navigation guidance, functionality explanation
• Software Interfaces: Feature identification, workflow assistance
• Visual Content: Diagrams, charts, images, design elements
• Error Messages: Troubleshooting, root cause analysis, solution recommendations

**RESPONSE GUIDELINES:**
✓ Address the specific query with relevant screen content insights
✓ For code: Explain logic, identify issues, suggest improvements
✓ For errors: Diagnose problems and provide actionable solutions
✓ For documents: Extract key information and provide analysis
✓ Maintain technical accuracy and professional communication
✓ Reference conversation history when contextually appropriate

✗ Do not interpret screen content as gestures or physical objects
✗ Avoid vague or overly general observations
✗ Do not provide unnecessary information beyond the query scope

Analyze the screen content and deliver your professional assessment."""
        
        response = gemini_client.generate_content([prompt, image])
        answer = response.text.strip()
        
        logger.info(f"Screen content analyzed: {answer[:100]}...")
        
        # Generate audio
        audio_base64 = text_to_speech(answer)
        
        # Store in history
        history_item = {
            "id": len(chat_history) + 1,
            "query": f"[Screen Share] {request.query}",
            "reasoning": "Analyzed screen share content",
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "audio": audio_base64
        }
        chat_history.append(history_item)
        save_chat_history()
        
        return {
            "reasoning": "Analyzed screen share content",
            "answer": answer,
            "audio": audio_base64
        }
        
    except Exception as e:
        logger.error(f"Error in screen content analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/continuous-sign-language")
async def continuous_sign_language(request: ContinuousSignRequest):
    """Handle continuous sign language detection from multiple frames"""
    if not gemini_client:
        raise HTTPException(status_code=500, detail="Gemini AI not configured")
    
    try:
        # Process multiple frames to understand the complete sign language sentence
        frames = []
        for frame_data in request.frames[-5:]:  # Use last 5 frames for context
            if ',' in frame_data:
                frame_data = frame_data.split(',', 1)[1]
            image_bytes = base64.b64decode(frame_data)
            frames.append(Image.open(io.BytesIO(image_bytes)))
        
        # Get conversation history for context (so AI remembers user's name, etc.)
        recent_history = get_recent_conversation_history(10)  # Last 10 conversations
        history_text = ""
        if recent_history:
            history_text = "\n\n**IMPORTANT CONTEXT FROM PREVIOUS CONVERSATION:**\n"
            for i in range(0, len(recent_history), 2):
                if i+1 < len(recent_history):
                    user_msg = recent_history[i].content[:150]
                    ai_msg = recent_history[i+1].content[:150]
                    history_text += f"• User said: {user_msg}\n• You replied: {ai_msg}\n"
            history_text += "\n**Use this context to answer their current question!**\n"
        
        # Enhanced prompt for sign language interpretation with ASL alphabet training
        prompt = f"""You are an expert AMERICAN SIGN LANGUAGE (ASL) INTERPRETER. You have been TRAINED on ASL alphabet and common signs.
{history_text}
**ASL ALPHABET REFERENCE** (for finger spelling):
A = Closed fist, thumb on side
B = Flat hand, fingers together, thumb across palm
C = Curved hand forming 'C' shape
D = Index finger up, other fingers touch thumb
E = Fingers curled into palm
F = Index and thumb form circle, other fingers up
G = Index finger and thumb point sideways
H = Index and middle finger extended horizontally
I = Pinky finger up, other fingers closed
J = Pinky traces 'J' in air
K = Index and middle finger up in 'V', thumb between them
L = Index finger up, thumb out forming 'L'
M = Thumb under three fingers
N = Thumb under two fingers
O = Fingers form circle
P = Like K but pointing down
Q = Index finger and thumb point down
R = Index and middle fingers crossed
S = Fist with thumb across fingers
T = Thumb between index and middle finger
U = Index and middle finger up together
V = Index and middle finger apart forming 'V'
W = Three fingers up (index, middle, ring)
X = Index finger crooked
Y = Thumb and pinky out (shaka sign)
Z = Index finger traces 'Z' in air

**COMMON ASL SIGNS:**
- HELLO = Wave hand
- THANK YOU = Hand from chin outward
- PLEASE = Circle on chest with flat hand
- YES = Fist nods like head
- NO = Index and middle finger snap closed with thumb
- SORRY = Fist circles on chest
- HOW ARE YOU = Point to person, then make 'A' shapes and raise them
- MY NAME = Point to self, then fingerspell name

**YOUR TASK:**
1. **RECOGNIZE** finger spelling across frames (they may spell words letter by letter)
2. **COMBINE** the letters into complete WORDS (W+H+O = "WHO", H+I = "HI")
3. **UNDERSTAND** the complete sentence/question they're asking
4. **RESPOND** naturally as if answering their question

**CRITICAL RULES:**
✅ DO: Combine finger-spelled letters into words (W-H-O A-M I = "WHO AM I?")
✅ DO: Answer their QUESTION, not describe the letters
✅ DO: Act like a human having a conversation
✅ DO: Respond with natural answers to their questions
✅ DO: Use information from conversation history (their name, previous topics, etc.)

❌ DON'T: Say "You're asking W+H+O A+M I?"
❌ DON'T: Just list the letters you see
❌ DON'T: Say "Based on the image..." or describe gestures
❌ DON'T: Say "Without further context..." or "I don't have capability to know"
❌ DON'T: Ignore conversation history when answering

**EXAMPLES:**
Frames show: H + I finger spelling
YOU RESPOND: "Hi! How can I help you?"
NOT: "You're signing H+I"

Frames show: W-H-O A-M I finger spelling (and history shows name is "Suresh Rawat")
YOU RESPOND: "You're Suresh Rawat! How can I help you today?"
NOT: "You're asking W+H+O A+M I?" or "I don't know who you are"

Frames show: H-E-L-L-O finger spelling
YOU RESPOND: "Hello! What can I do for you today?"
NOT: "You spelled H-E-L-L-O"

Frames show: T-H-A-N-K-S finger spelling
YOU RESPOND: "You're welcome! Glad I could help!"
NOT: "You're signing T-H-A-N-K-S"

Frames show: Waving hand
YOU RESPOND: "Hello! Nice to see you!"

Frames show: Thumbs up
YOU RESPOND: "Great! I'm glad you're happy. What's next?"

**Now analyze the frames, COMBINE the letters into words, and respond naturally to their message.**"""

        # Analyze with all frames for better context
        content = [prompt] + frames
        response = gemini_client.generate_content(content)
        answer = response.text.strip()
        
        logger.info(f"Sign language interpreted: {answer[:100]}...")
        
        # Generate audio response
        audio_base64 = text_to_speech(answer)
        
        # Store in history
        history_item = {
            "id": len(chat_history) + 1,
            "query": "[Sign Language]",
            "reasoning": "Continuous sign language interpretation",
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "audio": audio_base64
        }
        chat_history.append(history_item)
        save_chat_history()
        
        return {
            "reasoning": "Interpreted sign language across multiple frames",
            "answer": answer,
            "audio": audio_base64
        }
        
    except Exception as e:
        logger.error(f"Error in continuous sign language: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/ask/audio", response_model=QueryResponse)
async def ask_with_audio(request: AudioRequest):
    """Handle voice/audio input - transcribe and process"""
    if not gemini_client:
        raise HTTPException(status_code=500, detail="Gemini API key required for audio processing")
    
    try:
        # Transcribe audio to text
        transcribed_text = transcribe_audio(request.audio_data)
        
        if transcribed_text.startswith("Unable to transcribe"):
            return QueryResponse(
                reasoning="Audio transcription failed",
                answer=transcribed_text
            )
        
        # Combine transcribed text with optional additional query
        full_query = transcribed_text
        if request.query:
            full_query = f"{transcribed_text}. {request.query}"
        
        # Process through agent
        result = agent_executor.invoke({"input": full_query})
        
        reasoning = f"Voice input transcribed: '{transcribed_text}'. Used LangChain agent (Gemini)"
        if "intermediate_steps" in result and result["intermediate_steps"]:
            tools_used = [step[0].tool for step in result["intermediate_steps"]]
            reasoning += f" with tools: {', '.join(tools_used)}"
            reasoning += f" with tools: {', '.join(tools_used)}"
        
        answer = result.get("output", "No response generated")
        
        # Store in chat history
        history_item = {
            "id": len(chat_history) + 1,
            "query": f"[Voice] {transcribed_text}",
            "reasoning": reasoning,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
        chat_history.append(history_item)
        save_chat_history()  # Save to file
        
        return QueryResponse(reasoning=reasoning, answer=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/ask/image", response_model=QueryResponse)
async def ask_with_image(request: ImageRequest):
    """Handle image input - analyze or detect hand signs"""
    if not gemini_client:
        raise HTTPException(status_code=500, detail="Gemini API key required for image analysis")
    
    try:
        # Determine mode: analyze or hand_signs
        if request.mode == "hand_signs":
            analysis = detect_hand_signs(request.image_data)
            reasoning = "Analyzed image for hand signs/gestures using Gemini Vision"
            query_prefix = "[Hand Signs]"
        else:
            query = request.query or "What do you see in this image?"
            analysis = analyze_image(request.image_data, query)
            reasoning = f"Analyzed image using Gemini Vision with query: '{query}'"
            query_prefix = "[Image]"
        
        if analysis.startswith("Unable to"):
            return QueryResponse(reasoning=reasoning, answer=analysis)
        
        # If there's an additional text query, process through agent
        if request.query and request.mode == "analyze":
            full_input = f"Based on this image analysis: {analysis}. User question: {request.query}"
            result = agent_executor.invoke({"input": full_input})
            answer = result.get("output", analysis)
            reasoning += ". Then processed through LangChain agent (Gemini)"
        else:
            answer = analysis
        
        # Store in chat history
        history_item = {
            "id": len(chat_history) + 1,
            "query": f"{query_prefix} {request.query or 'Image analysis'}",
            "reasoning": reasoning,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
        chat_history.append(history_item)
        save_chat_history()  # Save to file
        
        return QueryResponse(reasoning=reasoning, answer=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/livekit/token")
async def get_livekit_token(room_name: str = "default-room", participant_name: str = "user"):
    """Generate LiveKit access token for WebRTC connection"""
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise HTTPException(status_code=500, detail="LiveKit credentials not configured")
    
    try:
        token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET) \
            .with_identity(participant_name) \
            .with_name(participant_name) \
            .with_grants(api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True
            ))
        
        return {
            "token": token.to_jwt(),
            "url": LIVEKIT_URL,
            "room": room_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating token: {str(e)}")

@app.post("/livekit/analyze-frame")
async def analyze_video_frame(file: UploadFile = File(...)):
    """Analyze a video frame for hand signs using Gemini Vision"""
    if not gemini_client:
        raise HTTPException(status_code=500, detail="Gemini AI not configured")
    
    try:
        # Read the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Prepare hand sign detection prompt
        prompt = """You are analyzing a video frame for HAND SIGNS and SIGN LANGUAGE.

Look at this image and:
1. Identify any hand signs or gestures you see
2. Describe the hand position and finger configuration
3. If it's a recognized sign (peace, thumbs up, OK, etc.), name it
4. If it's sign language, interpret the meaning if possible

Be specific about what you observe. If no clear hand sign is visible, say so clearly.

What hand sign or gesture do you see in this image?"""
        
        # Analyze with Gemini Vision
        response = gemini_client.generate_content([prompt, image])
        result = response.text.strip()
        
        logger.info(f"Video frame analyzed: {result[:100]}...")
        
        return {
            "analysis": result,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error analyzing video frame: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing frame: {str(e)}")

@app.post("/livekit/speak")
async def livekit_speak(text: str, room_name: str = "ai-assistant"):
    """
    Generate AI speech and send it to LiveKit room
    This endpoint generates TTS audio and publishes it to the LiveKit room
    """
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise HTTPException(status_code=500, detail="LiveKit credentials not configured")
    
    try:
        # Generate speech using gTTS
        logger.info(f"Generating LiveKit speech for room: {room_name}")
        audio_base64 = text_to_speech(text)
        
        # In a production app, you would:
        # 1. Create an AI bot participant
        # 2. Join the LiveKit room as a bot
        # 3. Publish audio track with the generated speech
        # 
        # For now, we'll return the audio and let the frontend play it
        # Full LiveKit bot integration requires the LiveKit server SDK
        
        return {
            "success": True,
            "audio": audio_base64,
            "message": "Audio generated successfully",
            "note": "Frontend will play audio through LiveKit room"
        }
    except Exception as e:
        logger.error(f"Error generating LiveKit speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.get("/history", response_model=List[ChatHistoryItem])
async def get_chat_history():
    """Get all chat history (temporary in-memory storage)"""
    return chat_history

@app.delete("/history")
async def clear_chat_history():
    """Clear all chat history"""
    chat_history.clear()
    save_chat_history()  # Save empty history to file
    return {"message": "Chat history cleared successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "langchain_enabled": agent_executor is not None,
        "multimodal_enabled": gemini_client is not None,
        "livekit_enabled": LIVEKIT_API_KEY is not None and LIVEKIT_API_SECRET is not None,
        "ai_provider": "Google Gemini (FREE)"
    }

# Mount static files (must be last)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Start server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
