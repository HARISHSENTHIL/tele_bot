import os
import asyncio
import logging
import time
import json
import redis
import torch
import openai
from io import BytesIO
from typing import Dict, Any, Optional, List
from PIL import Image
from diffusers import FluxPipeline
from datetime import datetime
import uuid

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Constants for image generation
LORA_ADAPTER_PATH = "./flux-lora-adapters-00/flux-lora-adapters-00.safetensors"
OCTOPUS_BRAND_PROMPT = (
    "cute orange octopus with white astronaut helmet. 2D line art with clean bold outlines, flat colors, procreate style, children book illustration "
    "The background should be relevant to the user input."
)

class ImageGenerationCoordinator:
    """
    Coordinates image generation requests across multiple bot instances
    using Redis for distributed queue management and load balancing.
    """
    
    def __init__(self):
        # Configure OpenAI
        if os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.openai_available = True
        else:
            logger.warning("OPENAI_API_KEY not set. Prompt enhancement will be disabled.")
            self.openai_available = False
        
        # Connect to Redis
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB,
            decode_responses=True  # For keys and non-binary values
        )
        
        # Keys for Redis
        self.queue_key = "image_gen:queue"
        self.active_key = "image_gen:active"
        self.stats_key = "image_gen:stats"
        self.instance_id = os.environ.get("INSTANCE_ID", f"instance-{uuid.uuid4().hex[:8]}")
        
        # Local tracking
        self.active_tasks = {}
        self.stats = {
            "total_successful": 0,
            "total_failed": 0,
            "active_generations": 0,
            "avg_generation_time": 0,
            "total_generation_time": 0,
            "generation_count": 0
        }
        
        # Initialize stats in Redis if not exists
        if not self.redis_client.exists(self.stats_key):
            self.redis_client.hset(self.stats_key, mapping=self.stats)
        
        # Configure H100 optimization settings
        self._setup_gpu_optimizations()
        
        # Load AI model
        self.pipe = self._load_model()
        
        # Start background task for processing the queue
        self.should_process_queue = True
        self.queue_processor_task = asyncio.create_task(self._process_queue())
    
    def _setup_gpu_optimizations(self):
        """Configure optimizations for H100 GPU."""
        if torch.cuda.is_available():
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
            logger.info(f"Using GPU: {gpu_name} with {gpu_mem:.2f} GB memory")
            
            # H100 specific settings
            if "H100" in gpu_name:
                logger.info("Detected H100 GPU - applying optimized settings")
                # Increase max concurrent tasks for H100
                settings.MAX_CONCURRENT_GENERATION = min(10, settings.MAX_CONCURRENT_GENERATION)
                # Use mixed precision for H100
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
    
    def _load_model(self):
        """Load the FLUX.1-dev model with LoRA adapters if available."""
        logger.info("Loading FLUX.1-dev model with LoRA adapters...")
        
        try:
            # H100 can handle bfloat16 precision well
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", 
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                variant="bf16",
                use_safetensors=True
            )
            
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
                # Enable memory efficient attention if available
                if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                    pipe.enable_xformers_memory_efficient_attention()
                
                # Use better attention implementation for H100
                if "H100" in torch.cuda.get_device_name(0):
                    if hasattr(pipe, 'unet'):
                        # Optimize attention mechanism for higher throughput
                        pipe.unet.set_attention_slice(1)
            
            if os.path.exists(LORA_ADAPTER_PATH):
                try:
                    logger.info(f"Loading LoRA adapter from {LORA_ADAPTER_PATH}")
                    pipe.load_lora_weights(LORA_ADAPTER_PATH)
                    logger.info("LoRA adapter loaded successfully")
                except Exception as lora_error:
                    logger.warning(f"Failed to load LoRA adapter: {lora_error}")
                    logger.warning("Continuing with base model only")
            else:
                logger.warning(f"LoRA adapter not found at {LORA_ADAPTER_PATH}, using base model")
                
            logger.info("FLUX.1-dev model loaded successfully")
            return pipe
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    async def close(self):
        """Clean up resources when shutting down."""
        self.should_process_queue = False
        if hasattr(self, 'queue_processor_task'):
            self.queue_processor_task.cancel()
            try:
                await self.queue_processor_task
            except asyncio.CancelledError:
                pass
        
        # Clean up any active tasks
        for task_id in list(self.active_tasks.keys()):
            self.redis_client.hdel(self.active_key, task_id)
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    async def generate_image(self, prompt: str, user_id: int) -> Optional[str]:
        """
        Queue an image generation request and wait for result.
        Returns the path to the generated image or None if failed.
        """
        # Create a unique task ID
        task_id = f"{self.instance_id}:{uuid.uuid4().hex}"
        
        # Create task data
        task_data = {
            "prompt": prompt,
            "user_id": user_id,
            "task_id": task_id,
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            "instance_id": self.instance_id
        }
        
        # Push to Redis queue
        self.redis_client.rpush(self.queue_key, json.dumps(task_data))
        logger.info(f"Queued task {task_id} for prompt: {prompt}")
        
        # Wait for result with timeout
        start_time = time.time()
        while time.time() - start_time < settings.GENERATION_TIMEOUT:
            # Check if this task has been processed
            task_result = self.redis_client.hget(self.active_key, task_id)
            if task_result:
                task_result = json.loads(task_result)
                if task_result.get("status") == "completed":
                    self.redis_client.hdel(self.active_key, task_id)
                    
                    # Update success stats
                    self._update_stats(
                        success=True, 
                        generation_time=time.time() - start_time
                    )
                    
                    return task_result.get("image_path")
                elif task_result.get("status") == "failed":
                    self.redis_client.hdel(self.active_key, task_id)
                    
                    # Update failure stats
                    self._update_stats(success=False)
                    
                    # Return None or raise exception based on error
                    error = task_result.get("error", "Unknown error")
                    logger.error(f"Task {task_id} failed: {error}")
                    return None
            
            # Wait before checking again
            await asyncio.sleep(0.5)
        
        # If we got here, the task timed out
        logger.warning(f"Task {task_id} timed out after {settings.GENERATION_TIMEOUT} seconds")
        
        # Update the task status to failed due to timeout
        self.redis_client.hset(
            self.active_key,
            task_id,
            json.dumps({"status": "failed", "error": "Timeout"})
        )
        
        # Update failure stats
        self._update_stats(success=False)
        
        return None
    
    async def _enhance_prompt(self, user_prompt):
        """Enhance the user prompt using OpenAI's API if available."""
        try:
            if not self.openai_available:
                logger.warning("OpenAI API key not set, skipping prompt enhancement")
                return user_prompt
                
            logger.info(f"Enhancing prompt: {user_prompt}")
            
            system_message = """
            You are a prompt enhancement expert for AI image generation models.
            Your task is to take a short user prompt and enhance it into a highly detailed, structured, and visually rich image prompt.
            Follow these enhancement rules:
            
            - **Scene Description**: Expand the setting and background details.
            - **Character Details (if applicable)**: Add specific attributes like clothing, facial expressions, accessories.
            - **Art Style**: Specify (anime, cyberpunk, watercolor, cinematic, 3D render, etc.).
            - **Lighting & Mood**: Describe the type of lighting and emotional tone.
            - **Composition Elements**: Add relevant camera angles, perspectives, and framing.
            - **Keyword Optimization**: Include essential words that improve image quality.

            Example:
            User Input: "octopus in Miami beach"
            Enhanced Output: "A vibrant and highly detailed digital illustration of an orange octopus wearing stylish sunglasses, a Hawaiian shirt, and a gold chain. The octopus is relaxing on Miami Beach, sipping a tropical cocktail. Its tentacles are animated—one adjusting its sunglasses, another holding a beach ball. The background features golden sand, palm trees, neon-lit beach bars, and luxury yachts in the turquoise ocean. The sunset casts a warm glow, creating a cinematic atmosphere with rich, colorful lighting."
            
            Now, enhance the following user prompt similarly.
            """
            
            response = openai.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Enhance this image prompt: {user_prompt}"}
                ],
                stream=False,
                max_tokens=140,
                temperature=0.7
            )
            
            enhanced_prompt = response.choices[0].message.content.strip()
            logger.info(f"Enhanced prompt: {enhanced_prompt}")
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Error enhancing prompt: {e}")
            return user_prompt
    
    async def _process_queue(self):
        """
        Background task that processes the image generation queue.
        Each bot instance runs this to handle generation requests.
        """
        logger.info(f"Started queue processor for instance {self.instance_id}")
        
        while self.should_process_queue:
            try:
                # Check if we have capacity to process more tasks
                current_active = len(self.active_tasks)
                if current_active >= settings.MAX_CONCURRENT_GENERATION:
                    await asyncio.sleep(0.5)
                    continue
                
                # Try to get a task from the queue
                task_json = self.redis_client.lpop(self.queue_key)
                if not task_json:
                    # No tasks in queue, wait a bit
                    await asyncio.sleep(0.5)
                    continue
                
                # Parse the task
                task_data = json.loads(task_json)
                task_id = task_data["task_id"]
                prompt = task_data["prompt"]
                
                # Update status to processing
                task_data["status"] = "processing"
                task_data["started_at"] = datetime.now().isoformat()
                task_data["processing_instance"] = self.instance_id
                
                # Add to active tasks
                self.redis_client.hset(self.active_key, task_id, json.dumps(task_data))
                self.active_tasks[task_id] = task_data
                
                # Update active count in stats
                await self._increment_active_count(1)
                
                # Start generation task
                generation_task = asyncio.create_task(
                    self._generate_image_task(task_id, prompt)
                )
                # We don't await this task - it runs in the background
                
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(1)  # Wait a bit before retrying
    
    async def _generate_image_task(self, task_id: str, prompt: str):
        """Process a single image generation task using FLUX.1-dev model."""
        try:
            logger.info(f"Generating image for task {task_id} with prompt: {prompt}")
            
            # Enhance the prompt using OpenAI
            enhanced_prompt = await self._enhance_prompt(prompt)
            
            # Check if the prompt contains references to an octopus
            lower_prompt = enhanced_prompt.lower()
            has_octopus = "octopus" in lower_prompt or "octa" in lower_prompt
            
            # Build the full prompt
            full_prompt = f"{OCTOPUS_BRAND_PROMPT}. {enhanced_prompt}"
            
            negative_prompt = "ugly, blurry, bad quality, distorted, deformed"
            
            # Generate the image - H100 optimized settings
            with torch.no_grad(), torch.cuda.amp.autocast():
                # H100 can handle larger images and more steps
                image = self.pipe(
                    prompt=full_prompt,
                    negative_prompt=negative_prompt,
                    height=1536,  # Higher resolution for H100
                    width=1536,   # Higher resolution for H100
                    guidance_scale=2,
                    num_inference_steps=25,  # More steps for better quality
                    generator=torch.Generator("cuda").manual_seed(int.from_bytes(os.urandom(4), "big"))
                ).images[0]
            
            # Save the image
            output_dir = os.path.join(os.getcwd(), "generated_images")
            os.makedirs(output_dir, exist_ok=True)
            
            image_filename = f"gen_{task_id.split(':')[-1]}.png"
            image_path = os.path.join(output_dir, image_filename)
            image.save(image_path, format="PNG", quality=100)  # Save at highest quality
            
            # Update task status to completed
            task_data = self.active_tasks.get(task_id, {})
            task_data["status"] = "completed"
            task_data["completed_at"] = datetime.now().isoformat()
            task_data["image_path"] = image_path
            task_data["enhanced_prompt"] = enhanced_prompt
            
            self.redis_client.hset(self.active_key, task_id, json.dumps(task_data))
            
            logger.info(f"Completed image generation for task {task_id}")
            
            # Clear CUDA cache after each generation to prevent memory fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error generating image for task {task_id}: {e}")
            
            # Update task status to failed
            task_data = self.active_tasks.get(task_id, {})
            task_data["status"] = "failed"
            task_data["error"] = str(e)
            
            self.redis_client.hset(self.active_key, task_id, json.dumps(task_data))
            
            # Clear CUDA cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        finally:
            # Remove from active tasks
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            # Decrement active count
            await self._increment_active_count(-1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics about image generation."""
        # Get the latest stats from Redis
        redis_stats = self.redis_client.hgetall(self.stats_key)
        
        # Get GPU memory stats if available
        gpu_stats = {}
        if torch.cuda.is_available():
            gpu_stats = {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}GB",
                "gpu_memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f}GB",
            }
        
        # Convert string values to appropriate types
        stats = {
            "active_generations": int(redis_stats.get("active_generations", 0)),
            "total_successful": int(redis_stats.get("total_successful", 0)),
            "total_failed": int(redis_stats.get("total_failed", 0)),
            "avg_generation_time": float(redis_stats.get("avg_generation_time", 0)),
            "instance_id": self.instance_id,
            **gpu_stats
        }
        
        return stats
    
    async def _increment_active_count(self, increment: int):
        """Update the count of active generations in Redis."""
        pipe = self.redis_client.pipeline()
        pipe.hincrby(self.stats_key, "active_generations", increment)
        pipe.execute()
    
    def _update_stats(self, success: bool, generation_time: float = 0):
        """Update generation statistics in Redis."""
        pipe = self.redis_client.pipeline()
        
        if success:
            pipe.hincrby(self.stats_key, "total_successful", 1)
            pipe.hincrby(self.stats_key, "generation_count", 1)
            pipe.hincrbyfloat(self.stats_key, "total_generation_time", generation_time)
            
            # Update average generation time
            gen_count = int(self.redis_client.hget(self.stats_key, "generation_count") or 0) + 1
            total_time = float(self.redis_client.hget(self.stats_key, "total_generation_time") or 0) + generation_time
            if gen_count > 0:
                avg_time = total_time / gen_count
                pipe.hset(self.stats_key, "avg_generation_time", avg_time)
        else:
            pipe.hincrby(self.stats_key, "total_failed", 1)
        
        pipe.execute() 