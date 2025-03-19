import os
import logging
import json
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, 
    ContextTypes, CallbackContext
)
import httpx
from dotenv import load_dotenv
from src.config import get_settings
from src.bot.handlers import register_handlers
from src.services.image_generator import ImageGenerationCoordinator

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Telegram Image Generation Bot")
settings = get_settings()

# Initialize the Telegram bot application
bot_app = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()

# Initialize image generation coordinator
image_coordinator = ImageGenerationCoordinator()

@app.on_event("startup")
async def on_startup():
    """Set webhook on startup."""
    webhook_url = settings.WEBHOOK_URL
    
    await bot_app.bot.set_webhook(
        url=webhook_url,
        secret_token=settings.WEBHOOK_SECRET
    )
    
    # Register all command and message handlers
    register_handlers(bot_app, image_coordinator)
    
    logger.info(f"Webhook set to {webhook_url}")
    logger.info("Bot started successfully")

@app.on_event("shutdown")
async def on_shutdown():
    """Remove webhook on shutdown."""
    await bot_app.bot.delete_webhook()
    await bot_app.shutdown()
    await image_coordinator.close()
    logger.info("Bot shutdown complete")

@app.post("/webhook")
async def webhook(request: Request):
    """Handle incoming webhook updates from Telegram."""
    # Verify secret token to ensure the request is from Telegram
    if request.headers.get("X-Telegram-Bot-Api-Secret-Token") != settings.WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    # Get the update data
    update_data = await request.json()
    logger.debug(f"Received update: {update_data}")
    
    # Create an Update object
    update = Update.de_json(data=update_data, bot=bot_app.bot)
    
    # Process the update
    await bot_app.process_update(update)
    
    return Response(status_code=200)

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancing and monitoring."""
    return {
        "status": "ok",
        "service": "telegram-image-bot",
        "version": "1.0.0",
        "instance_id": os.environ.get("INSTANCE_ID", "default")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app", 
        host=settings.HOST, 
        port=settings.PORT,
        reload=True
    ) 