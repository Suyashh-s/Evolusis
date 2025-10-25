"""
LiveKit Hand Sign Detection Bot
Connects to LiveKit room, receives video frames, detects hand signs with Gemini Vision
Responds in real-time when hand signs are detected
"""
import asyncio
import os
import base64
import io
import time
from typing import Optional
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from livekit import rtc
from livekit.api import AccessToken, VideoGrants
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
vision_model = genai.GenerativeModel('gemini-2.0-flash-exp')

class HandSignBot:
    def __init__(self, room_name: str = "ai-assistant"):
        self.room_name = room_name
        self.room: Optional[rtc.Room] = None
        self.last_analysis_time = 0
        self.analysis_interval = 3.0  # Analyze every 3 seconds
        self.running = False
        
    async def connect(self):
        """Connect to LiveKit room as AI bot"""
        logger.info(f"🤖 Connecting to room: {self.room_name}")
        
        # Generate token for bot
        token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        token = token.with_identity("ai-hand-sign-bot")
        token = token.with_name("AI Hand Sign Detector")
        token = token.with_grants(VideoGrants(
            room_join=True,
            room=self.room_name,
            can_subscribe=True,
            can_publish=True,
            can_publish_data=True
        ))
        
        # Create room and connect
        self.room = rtc.Room()
        self.running = True
        
        # Setup event handlers
        @self.room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            logger.info(f"👤 Participant joined: {participant.identity}")
            asyncio.create_task(self.send_message(
                f"👋 Hello! I'm the AI Hand Sign Detector. Show me hand signs and I'll identify them!",
                participant.identity
            ))
        
        @self.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            logger.info(f"📹 Track subscribed: {track.kind.name} from {participant.identity}")
            
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                logger.info("🎥 Starting hand sign detection...")
                asyncio.create_task(self.analyze_video_stream(track, participant))
        
        @self.room.on("track_unsubscribed")
        def on_track_unsubscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            logger.info(f"📹 Track unsubscribed: {track.kind.name} from {participant.identity}")
        
        @self.room.on("disconnected")
        def on_disconnected():
            logger.info("🔌 Disconnected from room")
            self.running = False
        
        # Connect to room
        await self.room.connect(LIVEKIT_URL, token.to_jwt())
        logger.info(f"✅ Bot connected to room: {self.room_name}")
        
        # Send welcome message to all participants
        await asyncio.sleep(1)  # Give time for connection to stabilize
        await self.send_message("🤖 **Hand Sign Detection Bot Online!**\n\nShow me hand signs and I'll identify them!")
        logger.info("📤 Sent welcome message")
        
    async def analyze_video_stream(self, track: rtc.VideoTrack, participant: rtc.RemoteParticipant):
        """Analyze video frames for hand signs"""
        logger.info("🔍 Video analysis started - watching for hand signs...")
        
        video_stream = rtc.VideoStream(track)
        frame_count = 0
        
        async for frame in video_stream:
            if not self.running:
                break
                
            frame_count += 1
            current_time = time.time()
            
            # Only analyze at intervals to avoid excessive API calls
            if current_time - self.last_analysis_time < self.analysis_interval:
                continue
                
            self.last_analysis_time = current_time
            
            try:
                logger.info(f"🖼️ Analyzing frame #{frame_count} for hand signs...")
                
                # Convert frame to PIL Image
                image = self.frame_to_image(frame)
                
                # Detect hand signs with Gemini
                result = await self.detect_hand_sign(image)
                
                if result:
                    logger.info(f"✋ Hand sign detected: {result[:100]}...")
                    # Broadcast to all participants
                    await self.send_message(result)
                else:
                    logger.info("👁️ No hand sign detected in this frame")
                    
            except Exception as e:
                logger.error(f"❌ Error analyzing frame: {e}", exc_info=True)
                
    def frame_to_image(self, frame: rtc.VideoFrame) -> Image.Image:
        """Convert LiveKit VideoFrame to PIL Image"""
        try:
            # Get frame data
            width = frame.width
            height = frame.height
            
            # Convert frame to argb
            argb_frame = frame.convert(rtc.VideoBufferType.RGBA)
            
            # Get buffer data
            buffer_data = bytes(argb_frame.data)
            
            # Create PIL Image
            image = Image.frombytes('RGBA', (width, height), buffer_data)
            
            return image
            
        except Exception as e:
            logger.error(f"❌ Error converting frame: {e}")
            raise
            
    async def detect_hand_sign(self, image: Image.Image) -> Optional[str]:
        """Detect hand signs in image using Gemini Vision"""
        try:
            # Prepare prompt specifically for hand sign detection
            prompt = """
You are an expert hand sign and gesture recognition AI. Analyze this image ONLY for hand signs and gestures.

YOUR TASK:
1. Look for hands making deliberate signs or gestures
2. If you see a clear hand sign, identify it and respond naturally
3. If you see no hands or no clear gesture, respond with exactly: "No hand sign detected"

COMMON HAND SIGNS TO RECOGNIZE:
- Peace sign (✌️): Two fingers up in V shape
- Thumbs up (👍): Thumb pointing up
- Thumbs down (👎): Thumb pointing down  
- OK sign (👌): Circle with thumb and index finger
- Wave (👋): Open palm, fingers spread
- Stop (✋): Open palm facing forward
- Fist (✊): Closed hand
- Pointing (☝️): Index finger extended
- Rock on (🤘): Index and pinky up, others folded
- I Love You (🤟): Thumb, index, and pinky up
- Number signs: Showing 1, 2, 3, 4, 5 fingers

RESPONSE FORMAT:
If hand sign detected:
"I see you're showing [sign name]! [Brief friendly comment about the gesture]"

If no hand sign:
"No hand sign detected"

BE CONVERSATIONAL AND FRIENDLY. Focus ONLY on hand signs - ignore other objects or backgrounds.
"""
            
            # Call Gemini Vision
            response = await asyncio.to_thread(
                vision_model.generate_content,
                [prompt, image]
            )
            
            result = response.text.strip()
            
            # Only return if hand sign was detected
            if "no hand sign detected" not in result.lower():
                return f"👋 **Hand Sign Detected!**\n\n{result}"
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Gemini Vision error: {e}")
            return None
            
    async def send_message(self, message: str, destination: Optional[str] = None):
        """Send message via LiveKit data channel"""
        try:
            data = message.encode('utf-8')
            
            if destination:
                # Send to specific participant
                await self.room.local_participant.publish_data(
                    data,
                    destination_identities=[destination],
                    reliable=True
                )
                logger.info(f"📤 Sent message to {destination}")
            else:
                # Broadcast to all
                await self.room.local_participant.publish_data(
                    data,
                    reliable=True
                )
                logger.info("📤 Broadcast message to all participants")
                
        except Exception as e:
            logger.error(f"❌ Error sending message: {e}")
            
    async def disconnect(self):
        """Disconnect from room"""
        self.running = False
        if self.room:
            await self.room.disconnect()
            logger.info("👋 Bot disconnected")

async def main():
    """Main entry point"""
    logger.info("🚀 Starting Hand Sign Detection Bot...")
    
    room_name = "ai-assistant"
    bot = HandSignBot(room_name)
    
    try:
        await bot.connect()
        
        # Keep bot running
        logger.info("✅ Bot is running and watching for hand signs...")
        logger.info("💡 Show hand signs to the camera and I'll identify them!")
        
        while bot.running:
            await asyncio.sleep(1)
        
    except KeyboardInterrupt:
        logger.info("⏹️ Stopping bot...")
    except Exception as e:
        logger.error(f"❌ Bot error: {e}", exc_info=True)
    finally:
        await bot.disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Bot stopped by user")
