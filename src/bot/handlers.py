import logging
import os
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    filters, ContextTypes
)
from src.services.image_generator import ImageGenerationCoordinator

logger = logging.getLogger(__name__)

# Command handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        f"Hi {user.mention_html()}! I'm an image generation bot. "
        f"Send me a text prompt, and I'll generate an image for you!\n\n"
        f"Use /help to see all available commands."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    help_text = (
        "Here are the available commands:\n\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/status - Check the bot's status\n\n"
        "To generate an image, simply send me a text description."
    )
    await update.message.reply_text(help_text)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check the status of the image generation service."""
    coordinator: ImageGenerationCoordinator = context.bot_data["image_coordinator"]
    stats = coordinator.get_stats()
    
    status_text = (
        f"🤖 Bot Status: Online\n"
        f"🖼️ Active Generations: {stats['active_generations']}\n"
        f"⏱️ Average Generation Time: {stats['avg_generation_time']:.2f}s\n"
        f"✅ Total Successful: {stats['total_successful']}\n"
        f"❌ Total Failed: {stats['total_failed']}\n"
        f"🔄 Instance ID: {stats['instance_id']}"
    )
    await update.message.reply_text(status_text)

# Message handlers
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages and generate images based on the prompt."""
    prompt = update.message.text
    user_id = update.effective_user.id
    user = update.effective_user
    user_name = user.username if user.username else user.full_name
    
    # Reference to the image generation coordinator
    coordinator: ImageGenerationCoordinator = context.bot_data["image_coordinator"]
    
    # Send initial response to user
    message = await update.message.reply_text("🎨 Processing your image generation request...")
    
    try:
        # Submit the generation task to the coordinator
        image_path = await coordinator.generate_image(prompt, user_id)
        
        if image_path:
            # Send the generated image back to the user
            caption = f"🐙 Here's your custom generated image! By: @{user_name} \n\n 🐙 Prompt: {prompt}"
            
            if len(caption) > 1000:
                caption = caption[:997] + "..."
                
            with open(image_path, 'rb') as photo_file:
                await update.message.reply_photo(
                    photo=photo_file,
                    caption=caption
                )
            await message.edit_text("✅ Image generated successfully!")
        else:
            # Handle case where generation failed but didn't raise an exception
            await message.edit_text("❌ Failed to generate image. Please try again later.")
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        await message.edit_text(
            f"❌ Error generating image: {str(e)}\n"
            f"Please try again with a different prompt or try later."
        )

def register_handlers(application: Application, image_coordinator: ImageGenerationCoordinator) -> None:
    """Register all handlers with the application."""
    # Store the image coordinator in the bot_data for access in handlers
    application.bot_data["image_coordinator"] = image_coordinator
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    
    # Add message handlers - anything that's not a command is treated as a generation prompt
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    # Log all errors
    application.add_error_handler(lambda update, context: logger.error(
        f"Exception while handling an update: {context.error}"
    )) 