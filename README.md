# Telegram Image Generation Bot

A scalable, load-balanced Telegram bot for image generation that can run on multiple instances. This bot uses the FLUX.1-dev model for high-quality image generation and OpenAI for prompt enhancement.

## Features

- **Webhook-Based**: Uses Telegram webhooks for faster processing
- **Multi-Instance Support**: Can run multiple instances for horizontal scaling
- **Load Balancing**: Distributes image generation tasks across all running instances
- **Redis-Based Queue**: Centralized task management with Redis
- **Health Monitoring**: Endpoints for checking bot health
- **Statistics**: Tracks performance metrics
- **FLUX.1-dev Model**: High-quality AI image generation
- **OpenAI Prompt Enhancement**: Improves user prompts for better image results
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs

## Prerequisites

- Python 3.8+
- Redis server
- CUDA-compatible GPU (recommended)
- An HTTPS domain with a valid SSL certificate (for webhooks)
- A Telegram Bot token from [@BotFather](https://t.me/BotFather)
- OpenAI API key (optional, for prompt enhancement)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd telegram-image-bot
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Download the FLUX.1-dev LoRA adapters:
   ```bash
   mkdir -p flux-lora-adapters-00
   wget -O flux-lora-adapters-00/flux-lora-adapters-00.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux-lora-adapters-00.safetensors
   ```

4. Copy the environment template and fill in your values:
   ```bash
   cp .env.example .env
   ```

## Configuration

Edit the `.env` file with your own values:

```
# Telegram Bot Settings
TELEGRAM_BOT_TOKEN=your_bot_token_here
WEBHOOK_URL=https://your-domain.com/webhook
WEBHOOK_SECRET=your_webhook_secret

# Server Settings
HOST=0.0.0.0
PORT=8000

# Redis Settings
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# Image Generation Settings
MAX_CONCURRENT_GENERATION=5
GENERATION_TIMEOUT=60

# OpenAI Settings (for prompt enhancement)
OPENAI_API_KEY=your_openai_api_key_here

# FLUX Model Settings
LORA_ADAPTER_PATH=./flux-lora-adapters-00/flux-lora-adapters-00.safetensors
USE_CUDA=true
```

## Running the Bot

### Local Development

For local development with a single instance:

```bash
python app.py
```

### Docker Deployment

For deployment with Docker and multiple instances:

```bash
docker-compose up -d
```

This will start:
- A Redis server for queue management
- Two bot instances with GPU support
- An Nginx load balancer

### Production Deployment

For production deployment with multiple instances:

1. Set up a Redis server for shared state management

2. Configure each instance with a unique `INSTANCE_ID` environment variable:
   ```bash
   INSTANCE_ID=instance-1 python app.py
   ```

3. Set up a reverse proxy (Nginx, etc.) to distribute traffic across instances

4. Use a process manager like Supervisor or systemd to manage the instances

## Deployment Architecture

```
                   +----------------+
                   |                |
                   | Telegram API   |
                   |                |
                   +--------+-------+
                            |
                            | Webhook
                            v
                   +----------------+
                   |                |
                   | Load Balancer  |
                   |                |
                   +--------+-------+
                            |
                +-----------+-----------+
                |                       |
  +-------------v---------+   +---------v-------------+
  |                       |   |                       |
  | Bot Instance 1        |   | Bot Instance 2        |
  | (FLUX.1-dev + GPU)    |   | (FLUX.1-dev + GPU)    |
  |                       |   |                       |
  +-------------+---------+   +---------+-------------+
                |                       |
                +----------+------------+
                           |
                 +---------v----------+
                 |                    |
                 | Redis Queue        |
                 |                    |
                 +--------------------+
```

## Image Generation Process

1. User sends a text prompt to the Telegram bot
2. The prompt is enhanced using OpenAI's GPT-4 (if API key is provided)
3. The enhanced prompt is queued in Redis
4. Available bot instances pick up tasks from the queue
5. Images are generated using the FLUX.1-dev model
6. Generated images are sent back to the user via Telegram

## Customizing the Image Generation

You can customize the image generation process by modifying:

- The prompt enhancement in `_enhance_prompt` method
- The base prompt in `OCTOPUS_BRAND_PROMPT` constant
- The image generation parameters in `_generate_image_task` method

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 