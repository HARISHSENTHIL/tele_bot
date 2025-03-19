#!/usr/bin/env python3
"""
Helper script to set up and verify your Telegram webhook.
"""

import argparse
import os
import requests
import json
import secrets
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_webhook_info(token):
    """Get current webhook information."""
    url = f"https://api.telegram.org/bot{token}/getWebhookInfo"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"ok": False, "error": f"Error {response.status_code}: {response.text}"}

def set_webhook(token, webhook_url, secret_token=None):
    """Set webhook for a Telegram bot."""
    url = f"https://api.telegram.org/bot{token}/setWebhook"
    
    data = {"url": webhook_url}
    if secret_token:
        data["secret_token"] = secret_token
    
    response = requests.post(url, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return {"ok": False, "error": f"Error {response.status_code}: {response.text}"}

def delete_webhook(token):
    """Delete the current webhook."""
    url = f"https://api.telegram.org/bot{token}/deleteWebhook"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"ok": False, "error": f"Error {response.status_code}: {response.text}"}

def generate_secret_token():
    """Generate a secure random secret token."""
    return secrets.token_hex(16)

def get_ngrok_url():
    """Try to get the Ngrok tunnel URL if running."""
    try:
        response = requests.get("http://127.0.0.1:4040/api/tunnels")
        if response.status_code == 200:
            data = response.json()
            for tunnel in data["tunnels"]:
                if tunnel["proto"] == "https":
                    return tunnel["public_url"]
        return None
    except:
        return None

def update_env_file(webhook_url, secret_token):
    """Update the .env file with the webhook information."""
    # Read current .env file
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            lines = f.readlines()
    else:
        lines = []
    
    # Find and update or add webhook variables
    webhook_url_found = False
    webhook_secret_found = False
    new_lines = []
    
    for line in lines:
        if line.strip().startswith("WEBHOOK_URL="):
            new_lines.append(f"WEBHOOK_URL={webhook_url}\n")
            webhook_url_found = True
        elif line.strip().startswith("WEBHOOK_SECRET="):
            new_lines.append(f"WEBHOOK_SECRET={secret_token}\n")
            webhook_secret_found = True
        else:
            new_lines.append(line)
    
    # Add missing variables
    if not webhook_url_found:
        new_lines.append(f"WEBHOOK_URL={webhook_url}\n")
    if not webhook_secret_found:
        new_lines.append(f"WEBHOOK_SECRET={secret_token}\n")
    
    # Write back to .env file
    with open(".env", "w") as f:
        f.writelines(new_lines)

def main():
    parser = argparse.ArgumentParser(description="Set up or check a Telegram bot webhook.")
    parser.add_argument("--token", help="Telegram bot token")
    parser.add_argument("--url", help="Webhook URL (include /webhook endpoint)")
    parser.add_argument("--info", action="store_true", help="Get current webhook info")
    parser.add_argument("--delete", action="store_true", help="Delete current webhook")
    parser.add_argument("--update-env", action="store_true", help="Update .env file with webhook info")
    
    args = parser.parse_args()
    
    # Get token from args or environment
    token = args.token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: No Telegram bot token provided. Use --token or set TELEGRAM_BOT_TOKEN in .env")
        return
    
    # Just get info if requested
    if args.info:
        info = get_webhook_info(token)
        print(json.dumps(info, indent=2))
        return
    
    # Delete webhook if requested
    if args.delete:
        result = delete_webhook(token)
        print(json.dumps(result, indent=2))
        return
    
    # Try to get webhook URL
    webhook_url = args.url or os.getenv("WEBHOOK_URL")
    
    # If no URL provided, try to detect Ngrok
    if not webhook_url:
        ngrok_url = get_ngrok_url()
        if ngrok_url:
            webhook_url = f"{ngrok_url}/webhook"
            print(f"Detected Ngrok tunnel. Using URL: {webhook_url}")
        else:
            print("Error: No webhook URL provided. Use --url or set WEBHOOK_URL in .env")
            return
    
    # Make sure URL ends with /webhook
    if not webhook_url.endswith("/webhook"):
        webhook_url = webhook_url.rstrip("/") + "/webhook"
    
    # Get or generate webhook secret
    webhook_secret = os.getenv("WEBHOOK_SECRET")
    if not webhook_secret:
        webhook_secret = generate_secret_token()
        print(f"Generated new webhook secret: {webhook_secret}")
    
    # Set the webhook
    result = set_webhook(token, webhook_url, webhook_secret)
    print(json.dumps(result, indent=2))
    
    if result.get("ok", False) and args.update_env:
        update_env_file(webhook_url, webhook_secret)
        print(f"Updated .env file with webhook URL and secret.")

if __name__ == "__main__":
    main() 