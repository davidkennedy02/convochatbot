import os
import argparse
import sys
import torch
import pandas as pd
import numpy as np
from convokit import Corpus, Utterance, Speaker
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, 
                          TrainingArguments, DataCollatorForLanguageModeling)
from torch.utils.data import Dataset
import shutil
import logging
import re
import json
from datetime import datetime
import gc  # For garbage collection
import time
import random
import traceback
from tqdm import tqdm  # For progress bars

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot_log.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Memory management function
def manage_memory():
    """Force garbage collection and clear CUDA cache if available"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Define debug mode flag - don't use global keyword outside functions
DEBUG_MODE = False

# Custom dataset class to prepare training examples
class ChatDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        # Concatenate context and response with the EOS token as a separator.
        text = example["context"] + self.tokenizer.eos_token + example["response"] + self.tokenizer.eos_token
        tokenized = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        # Squeeze to remove extra batch dimension.
        input_ids = tokenized.input_ids.squeeze()
        attention_mask = tokenized.attention_mask.squeeze()  # Add attention mask
        # For causal LM, the labels are the same as input_ids.
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove URLs
    text = re.sub(r'http\S+', '[URL]', text)
    # Replace emails
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    return text

# Function to validate CSV before processing
def validate_csv(csv_path):
    """Validate that the CSV file exists and contains required columns"""
    try:
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return False
            
        # Try to read a small sample to validate format
        df_sample = pd.read_csv(csv_path, nrows=5)
        
        # Check for required columns
        if 'conversation_id' not in df_sample.columns:
            logger.error("CSV must contain 'conversation_id' column")
            return False
        
        if 'message' not in df_sample.columns and 'text' not in df_sample.columns:
            logger.error("CSV must contain either 'message' or 'text' column")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating CSV file: {e}")
        return False

# Function to load and process data from CSV using convokit.
def load_data(csv_path, tokenizer, context_window=3, max_length=512, max_samples=None, sample_frac=0.05):
    """
    Load and process conversation data from CSV
    
    Args:
        csv_path: Path to the CSV file
        tokenizer: Tokenizer for encoding text
        context_window: Number of previous messages to include as context
        max_length: Maximum length of tokenized sequences
        max_samples: Optional limit on number of examples to create
        sample_frac: Fraction of data to sample
    """
    # First validate the CSV
    if not validate_csv(csv_path):
        raise ValueError(f"Invalid CSV file: {csv_path}")
    
    logger.info(f"Loading data from {csv_path}, sampling {sample_frac*100}% of data...")
    
    # Load the CSV with pandas.
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded CSV with {len(df)} rows")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise

    # Use only specified fraction of the data
    original_size = len(df)
    df = df.sample(frac=sample_frac, random_state=42)
    logger.info(f"Sampled {len(df)} rows from {original_size} ({sample_frac*100}%)")
    
    # Debug output for DataFrame columns - use the module-level DEBUG_MODE directly
    if DEBUG_MODE:
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        logger.info(f"First row: {df.iloc[0].to_dict()}")
    
    # Ensure conversation_id is a string.
    df['conversation_id'] = df['conversation_id'].astype(str)
    
    # Create a unique utterance_id using the row index.
    df['utterance_id'] = df.index.astype(str)
    
    # Alternate speakers using cumcount.
    df['speaker'] = df.groupby('conversation_id').cumcount().apply(lambda x: "speaker1" if x % 2 == 0 else "speaker2")
    
    # Check and rename 'message' column to 'text' for Convokit if it exists
    if 'message' in df.columns:
        df = df.rename(columns={'message': 'text'})
    elif 'text' not in df.columns:
        logger.error("Neither 'message' nor 'text' column found in CSV")
        raise ValueError("CSV must contain either 'message' or 'text' column")
    
    # Check if 'sentiment' column exists, otherwise add a default value
    if 'sentiment' not in df.columns:
        logger.warning("No 'sentiment' column found in CSV, adding default neutral sentiment")
        df['sentiment'] = 'neutral'

    # Check for empty or null text values
    empty_rows = df['text'].isna().sum()
    if empty_rows > 0:
        logger.warning(f"Found {empty_rows} rows with empty text values. Filling with placeholder.")
        df['text'] = df['text'].fillna("No message content")
    
    # Create Utterance objects.
    utterances = []
    # Dictionary to store Speaker objects per conversation.
    speaker_dict = {}
    
    # Preprocess text before creating utterances
    logger.info("Creating utterances...")
    try:
        for row in df.itertuples(index=False):
            # Create a unique key per conversation and speaker.
            spk_key = (row.conversation_id, row.speaker)
            if spk_key not in speaker_dict:
                speaker_dict[spk_key] = Speaker(id=row.speaker)
            
            # Preprocess the text
            processed_text = preprocess_text(row.text)
            
            utt = Utterance(
                id=row.utterance_id,
                conversation_id=row.conversation_id,
                speaker=speaker_dict[spk_key],
                text=processed_text,
                meta={"sentiment": row.sentiment}
            )
            utterances.append(utt)
    except Exception as e:
        logger.error(f"Error creating utterances: {e}")
        traceback.print_exc()
        raise
    
    # Build the Corpus from utterances.
    logger.info(f"Building corpus from {len(utterances)} utterances...")
    try:
        corpus = Corpus(utterances=utterances)
        logger.info(f"Corpus built with {len(corpus.get_conversation_ids())} conversations")
    except Exception as e:
        logger.error(f"Error building corpus: {e}")
        traceback.print_exc()
        raise
    
    # IMPROVEMENT: Add batched processing for large datasets
    batch_size = min(1000, max(10, len(corpus.get_conversation_ids())))
    all_examples = []
    
    # Process conversations in batches
    logger.info("Processing conversations...")
    try:
        for batch_idx in tqdm(range(0, len(corpus.get_conversation_ids()), batch_size), desc="Processing conversations"):
            batch_ids = list(corpus.get_conversation_ids())[batch_idx:batch_idx+batch_size]
            batch_examples = []
            
            for conv_id in batch_ids:
                conv = corpus.get_conversation(conv_id)
                # Retrieve utterances using the utterance IDs and the get_utterance method.
                conv_utts = sorted(
                    [conv.get_utterance(utt_id) for utt_id in conv._utterance_ids],
                    key=lambda utt: int(utt.id)
                )
                
                messages = [utt.text for utt in conv_utts]
                for msg_idx in range(len(messages)):
                    # Dynamic context window with smarter handling
                    context_messages = []
                    total_length = len(tokenizer.encode(messages[msg_idx] + tokenizer.eos_token))
                    
                    # Start with most recent message and go back as far as possible
                    for j in range(msg_idx-1, max(-1, msg_idx-context_window-1), -1):
                        message_tokens = tokenizer.encode(messages[j])
                        if total_length + len(message_tokens) < max_length:
                            context_messages.insert(0, messages[j])
                            total_length += len(message_tokens)
                        else:
                            break
                    
                    context = " ".join(context_messages)
                    response = messages[msg_idx]
                    batch_examples.append({"context": context, "response": response})
            
            all_examples.extend(batch_examples)
            manage_memory()  # Clean up memory after each batch
            
            if max_samples and len(all_examples) >= max_samples:
                all_examples = all_examples[:max_samples]
                logger.info(f"Reached max samples limit ({max_samples})")
                break
    except Exception as e:
        logger.error(f"Error processing conversations: {e}")
        traceback.print_exc()
        raise
        
    logger.info(f"Created {len(all_examples)} training examples")
    
    # Debug: Show sample examples
    if DEBUG_MODE and all_examples:
        logger.info(f"Sample example - Context: '{all_examples[0]['context']}'")
        logger.info(f"Sample example - Response: '{all_examples[0]['response']}'")

    return all_examples

# Function to test model loading
def test_model_loading(model_name):
    """Test if the model can be loaded successfully"""
    try:
        logger.info(f"Testing model loading: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        logger.info(f"Model {model_name} loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        return False

# Function to train the chatbot model
def train_model(csv_path, model_name="microsoft/DialoGPT-small", output_dir="./chatbotmodel", 
                context_window=3, epochs=3, resume_training=False, max_samples=None, sample_frac=0.05,
                debug=False):
    """Train or fine-tune a chatbot model"""
    global DEBUG_MODE  # Properly declare we want to modify the global variable
    DEBUG_MODE = debug
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Test model loading first
    if not test_model_loading(model_name):
        raise ValueError(f"Failed to load model {model_name}. Please check your internet connection or model name.")
    
    # Load the tokenizer and model from transformers.
    try:
        logger.info(f"Loading tokenizer and model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Check model was loaded properly
        if not hasattr(model, 'generate'):
            logger.error("Loaded model doesn't have a generate method. Incorrect model type?")
            raise ValueError("Loaded model doesn't have expected methods")
            
        logger.info(f"Moving model to {device}...")
        model.to(device)
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        traceback.print_exc()
        raise
    
    logger.info("Adding special tokens...")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    # Load and process the data.
    logger.info("Loading and processing data...")
    try:
        examples = load_data(csv_path, tokenizer, context_window, 
                            max_length=tokenizer.model_max_length, 
                            max_samples=max_samples,
                            sample_frac=sample_frac)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        traceback.print_exc()
        raise
    
    # Split into training and validation sets
    from sklearn.model_selection import train_test_split
    logger.info("Splitting data into train and validation sets...")
    train_examples, val_examples = train_test_split(examples, test_size=0.1, random_state=42)
    
    logger.info(f"Training examples: {len(train_examples)}")
    logger.info(f"Validation examples: {len(val_examples)}")
    
    # Check if we have enough data
    if len(train_examples) < 10:
        logger.warning("Very small training dataset. Results may be poor.")
        if len(train_examples) == 0:
            raise ValueError("No training examples found. Check your CSV file and preprocessing.")
    
    logger.info("Creating datasets...")
    try:
        train_dataset = ChatDataset(train_examples, tokenizer)
        val_dataset = ChatDataset(val_examples, tokenizer)
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        traceback.print_exc()
        raise
    
    # Set up a data collator for causal LM (no masked language modeling).
    logger.info("Creating data collator...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Create unique run name with timestamp
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    
    # Check if we want to resume training from a checkpoint
    starting_epoch = 0
    if resume_training and os.path.exists(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint")]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints)[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
            starting_epoch = int(latest_checkpoint.split("-")[1])
    
    # Calculate batch size based on available GPU memory
    batch_size = 4
    if torch.cuda.is_available():
        # Use a smaller batch size for larger models
        if "large" in model_name.lower():
            batch_size = 2
        elif "medium" in model_name.lower():
            batch_size = 3
        elif "small" in model_name.lower():
            batch_size = 4
        else:
            batch_size = 2  # Default to smaller size for unknown models
            
    logger.info(f"Using batch size: {batch_size}")
    
    # Set training arguments with more configuration options
    try:
        logger.info("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,  
            per_device_eval_batch_size=batch_size,
            eval_steps=100,
            save_steps=100,
            save_total_limit=3,
            logging_steps=10,  # More frequent logging
            evaluation_strategy="steps",
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=100,  # Adjusted for smaller datasets
            prediction_loss_only=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",  # Disable wandb, etc.
            run_name=run_name,
            gradient_accumulation_steps=4,  # For handling larger batch sizes effectively
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            dataloader_num_workers=0,  # Set to 0 for debugging
            dataloader_pin_memory=torch.cuda.is_available(),  # Speed up data transfer to GPU
        )
    except Exception as e:
        logger.error(f"Error setting up training arguments: {e}")
        traceback.print_exc()
        raise
    
    # Initialize the Trainer.
    try:
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
    except Exception as e:
        logger.error(f"Error initializing trainer: {e}")
        traceback.print_exc()
        raise
    
    try:
        # Start training.
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=resume_training)
        
        # Evaluate the model
        logger.info("Evaluating model...")
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        
        # Save the fine-tuned model and tokenizer.
        logger.info(f"Saving model to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save model metadata
        metadata = {
            "base_model": model_name,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": epochs,
            "eval_perplexity": float(np.exp(eval_results["eval_loss"])),
            "context_window": context_window,
            "training_examples": len(train_examples)
        }
        
        with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f)
        
        logger.info("Training completed successfully!")    
        return model, tokenizer
        
    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving current model state.")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error during training: {e}")
        traceback.print_exc()
        # Don't save potentially corrupted model
        raise

# Add a dry run function
def dry_run(args):
    """Do a dry run to validate setup without training"""
    logger.info("Starting dry run...")
    
    # Test model loading
    if not test_model_loading(args.model_name):
        logger.error(f"Failed to load model {args.model_name}")
        return False
        
    # Test CSV loading
    if not validate_csv(args.csv):
        logger.error(f"Failed to validate CSV file {args.csv}")
        return False
    
    # Load tokenizer for data processing test
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Try processing a small sample
    try:
        examples = load_data(args.csv, tokenizer, context_window=args.context_window, 
                            max_samples=5, sample_frac=0.001)
        logger.info(f"Successfully processed {len(examples)} examples")
    except Exception as e:
        logger.error(f"Failed to process data: {e}")
        return False
    
    logger.info("Dry run successful! Your setup appears valid.")
    return True

# Interactive chat loop.
def chat_loop(model, tokenizer):
    """
    Interactive chat loop with the model
    
    Args:
        model: The language model to use
        tokenizer: The tokenizer for the model
    """
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Put model in evaluation mode and disable gradient computation for inference
    model.eval()
    torch.set_grad_enabled(False)
    
    logger.info(f"Using device for inference: {device}")
    
    print("Chatbot is ready! Type 'exit' or 'quit' to end the conversation.")
    print("Special commands: /clear (clear history), /save (save conversation), /load (load conversation)")
    
    chat_history = []
    max_history_tokens = min(512, model.config.max_position_embeddings - 100)  # Allow more room for response
    conversation_directory = "./conversations"
    os.makedirs(conversation_directory, exist_ok=True)
    
    # Improved fallback responses with topic continuation capability
    fallback_responses = [
        "I'd like to understand more about that. Could you elaborate?",
        "That's an interesting point. What aspects of that do you enjoy most?",
        "I'm curious to hear more about your thoughts on that subject.",
        "Let's explore that topic further. What specifically interests you about it?",
        "That's something I'd like to discuss more. Could you share more details?"
    ]
    
    # System prompt to guide model behavior
    system_prompt = "You are a helpful and friendly assistant engaged in a casual conversation. Your responses are coherent, contextually relevant, and natural-sounding. When the user mentions a topic, you engage with that topic in a thoughtful way."
    
    while True:
        user_input = input("User: ")
        
        # Add sanitization for user input
        user_input = user_input.strip()
        if not user_input:
            print("Please enter a message.")
            continue
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # Special commands
        if user_input.lower() == "/clear":
            chat_history = []
            print("Conversation history cleared.")
            continue
            
        elif user_input.lower().startswith("/save"):
            parts = user_input.split()
            filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            if len(parts) > 1:
                filename = f"{parts[1]}.txt"
            
            with open(os.path.join(conversation_directory, filename), "w") as f:
                f.write("\n".join(chat_history))
            print(f"Conversation saved to {filename}")
            continue
            
        elif user_input.lower().startswith("/load"):
            parts = user_input.split()
            if len(parts) == 1:
                print("Please specify a file to load.")
                continue
                
            filename = f"{parts[1]}.txt"
            try:
                with open(os.path.join(conversation_directory, filename), "r") as f:
                    chat_history = f.read().splitlines()
                print(f"Loaded conversation from {filename}")
            except FileNotFoundError:
                print(f"File {filename} not found.")
            continue

        # Add user input to chat history
        chat_history.append(f"User: {user_input}")
        
        # Extract last topic if available from previous exchanges
        last_topic = ""
        if len(chat_history) >= 3:  # We have at least one complete exchange plus current user input
            last_bot_response = chat_history[-2].replace("Bot: ", "")
            last_user_input = chat_history[-1].replace("User: ", "")
            # Try to identify key nouns as topics using simple approach
            potential_topics = set()
            for msg in [last_bot_response, last_user_input]:
                words = msg.split()
                # Look for capitalized words or words following "the", "a", "an" as potential nouns
                for i, word in enumerate(words):
                    if (word[0].isupper() and i > 0) or \
                       (i > 0 and words[i-1].lower() in ["the", "a", "an"]):
                        potential_topics.add(word.strip(".,!?").lower())
            if potential_topics:
                last_topic = f" The conversation mentions: {', '.join(potential_topics)}."
        
        # Prepare the input for the model with improved prompting
        if len(chat_history) > 1:
            # Use more context for better coherence - last 4 exchanges
            context = chat_history[-min(len(chat_history), 8):]  # Increased from 5 to 8
            
            # Format the prompt with stronger guidance
            formatted_context = "\n".join(context)
            full_prompt = (f"{system_prompt}{last_topic}\n\n"
                          f"Conversation History:\n{formatted_context}\n\n"
                          f"Please provide a natural, helpful response to the user's latest message.")
        else:
            full_prompt = f"{system_prompt}\n\nUser: {user_input}\n\nPlease respond to the user in a helpful and natural way."
        
        # Ensure we don't exceed model's max length by truncating history if needed
        encoded_prompt = tokenizer.encode(full_prompt, return_tensors="pt")[0]
        if len(encoded_prompt) > max_history_tokens:
            # Try to find sentence boundaries to truncate naturally
            while len(encoded_prompt) > max_history_tokens:
                chat_history.pop(0)  # Remove oldest message
                context = chat_history[-min(len(chat_history), 8)]
                formatted_context = "\n".join(context)
                full_prompt = (f"{system_prompt}{last_topic}\n\n"
                              f"Conversation History:\n{formatted_context}\n\n"
                              f"Please provide a natural, helpful response to the user's latest message.")
                encoded_prompt = tokenizer.encode(full_prompt, return_tensors="pt")[0]
        
        input_ids = encoded_prompt.unsqueeze(0).to(device)
        
        # Generate a response with improved parameters
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Generate with adjusted parameters for more coherent responses
                start_time = time.time()
                output = model.generate(
                    input_ids,
                    max_new_tokens=150,  # Increased for more complete responses
                    pad_token_id=tokenizer.pad_token_id,
                    temperature=0.7,  # Slightly increased for more variety
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.2,
                    do_sample=True,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    num_beams=3,  # Increased for better coherence
                    bad_words_ids=[[tokenizer.unk_token_id]] if hasattr(tokenizer, 'unk_token_id') else None
                )
                
                # Check for extremely slow generation
                if time.time() - start_time > 10:
                    logger.warning("Response generation took too long")
                
                # Get only the newly generated tokens
                response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
                
                # Clean up the response
                response = response.strip()
                
                # Extract actual response from generated text
                # Look for the response after the formatted prompt
                response_markers = ["Bot:", "Assistant:", "Response:", "Reply:"]
                extracted = False
                for marker in response_markers:
                    if marker in response:
                        parts = response.split(marker, 1)
                        if len(parts) > 1:
                            response = parts[1].strip()
                            extracted = True
                            break
                
                if not extracted:
                    # If no marker found, try to get the first paragraph
                    paragraphs = response.split('\n')
                    if paragraphs:
                        response = paragraphs[0].strip()
                
                # Additional cleanup
                response = re.sub(r'<.*?>', '', response)  # Remove any HTML-like tags
                
                # Remove any common assistant identifiers
                response = re.sub(r'(?i)(assistant|AI):\s*', '', response)
                
                # Stop at any turn-taking markers
                stop_phrases = ["User:", "Human:", "Person:", "You:", "Question:"]
                for phrase in stop_phrases:
                    if phrase in response:
                        response = response.split(phrase)[0].strip()
                
                # Check response quality with improved assessment
                response_quality = assess_response_quality(response, user_input)
                
                if response_quality == "poor" or not response:
                    # Create a more contextual fallback response
                    # Extract potential topic from user input for better fallback
                    words = user_input.split()
                    nouns = [word for word in words if len(word) > 3 and word.lower() not in 
                             ["what", "when", "where", "which", "who", "whom", "whose", "why", "how"]]
                    
                    if nouns and random.random() < 0.7:  # 70% chance to use topic-specific fallback
                        topic = random.choice(nouns).strip(".,!?")
                        fallback = f"I'd like to hear more about {topic}. Could you tell me more about what interests you about it?"
                    else:
                        fallback = random.choice(fallback_responses)
                        
                    logger.warning(f"Generated poor quality response, using fallback")
                    response = fallback
                    
                print(f"Bot: {response}")
                chat_history.append(f"Bot: {response}")
                break
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"CUDA OOM, attempt {attempt + 1}/{max_retries}")
                    manage_memory()
                    continue
                logger.error(f"Error generating response: {e}")
                print("Bot: I'm having technical difficulties. Let's try again with a simpler question.")
                break
                
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                print("Bot: Sorry, I couldn't generate a response right now.")
                break

def assess_response_quality(response, user_input):
    """
    Assess the quality of a generated response
    
    Args:
        response: The text response to evaluate
        user_input: The user's input for context
        
    Returns:
        quality: String indicating 'good', 'medium', or 'poor' quality
    """
    # Empty or very short responses
    if not response or len(response) < 5:
        return "poor"
    
    # Check for common nonsensical patterns
    if re.search(r'([a-zA-Z])\1{3,}', response):  # Repeated characters
        return "poor"
        
    # Check for word salad (many random words together without proper sentences)
    words = response.split()
    if len(words) > 8:
        # Count capital letters in unusual places
        mid_caps = sum(1 for i, word in enumerate(words[1:], 1) 
                        if word[0].isupper() and words[i-1][-1] not in '.!?')
        if mid_caps > 3:
            return "poor"
    
    # Check for too many special characters
    special_char_ratio = sum(1 for c in response if not c.isalnum() and not c.isspace()) / len(response) if response else 0
    if special_char_ratio > 0.25:
        return "poor"
    
    # Check for overly repetitive content
    word_counter = {}
    for word in words:
        word_counter[word.lower()] = word_counter.get(word.lower(), 0) + 1
    
    max_repetitions = max(word_counter.values()) if word_counter else 0
    if max_repetitions > 3 and len(words) > 10:
        return "poor"
        
    # New: Check for topical relevance to user input
    user_words = set(user_input.lower().split())
    response_words = set(response.lower().split())
    
    # Filter out common stop words
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", 
                 "about", "is", "are", "am", "was", "were", "be", "been", "have", "has", "had",
                 "do", "does", "did", "can", "could", "will", "would", "should", "may", "might",
                 "must", "that", "this", "these", "those", "i", "you", "he", "she", "it", "we", "they"}
    
    user_content_words = {w for w in user_words if w not in stop_words and len(w) > 2}
    response_content_words = {w for w in response_words if w not in stop_words and len(w) > 2}
    
    # If user input has content words but none are reflected in response, consider it less relevant
    if (len(user_content_words) >= 2 and len(response_content_words) >= 3 and 
        not any(w in response_content_words for w in user_content_words)):
        # Allow if it's a question (the response asks for clarification)
        if '?' in response and len(response) < 80:
            return "good"
        return "medium"  # Don't immediately fail but consider it medium quality
        
    # Check for nonsensical short responses with generic words
    generic_responses = {"i am", "it is", "that is", "they are", "we are"}
    if len(response) < 20 and any(phrase in response.lower() for phrase in generic_responses):
        # Check if it's just a generic statement without context
        if not any(w in response_content_words for w in user_content_words):
            return "poor"
    
    # If it passed all checks, it's probably good
    return "good"

# Main entry point.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and run a chatbot.")
    parser.add_argument("--train", action="store_true", help="Train the model on the dataset")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without training")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with extra logging")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--csv", type=str, default="topical_chat.csv", help="Path to the dataset CSV file")
    parser.add_argument("--output_dir", type=str, default="./chatbotmodel", help="Directory to save/load the model")
    parser.add_argument("--context_window", type=int, default=3, help="Number of previous utterances to include")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--model_name", type=str, default="microsoft/DialoGPT-small", help="Base model to fine-tune")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of training examples to use")
    parser.add_argument("--sample_frac", type=float, default=0.05, help="Fraction of data to sample (default: 5%)")
    parser.add_argument("--chat_only", action="store_true", help="Skip loading pretrained model and just use base model")
    args = parser.parse_args()

    try:
        # Create requirements check function
        def check_requirements():
            """Check for critical requirements and dependencies"""
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                logger.warning("CUDA not available, using CPU. Training will be slow.")
            
            # Check for critical packages
            try:
                import transformers
                logger.info(f"Transformers version: {transformers.__version__}")
                import pandas
                logger.info(f"Pandas version: {pandas.__version__}")
                import numpy
                logger.info(f"NumPy version: {numpy.__version__}")
            except ImportError as e:
                logger.error(f"Missing required package: {e}")
                print(f"Please install the missing package: {e}")
                sys.exit(1)

            return True

        check_requirements()
        
        # Run in debug mode if requested - don't use global keyword here
        DEBUG_MODE = args.debug  # Just assign directly to the module-level variable
        if DEBUG_MODE:
            logger.setLevel(logging.DEBUG)
            logger.info("Debug mode enabled")
        
        # Run dry run if requested
        if args.dry_run:
            if dry_run(args):
                sys.exit(0)
            else:
                logger.error("Dry run failed. Please fix the issues before training.")
                sys.exit(1)
        
        # Chat-only mode - just use the base model without fine-tuning
        if args.chat_only:
            logger.info(f"Chat-only mode with base model: {args.model_name}")
            model = AutoModelForCausalLM.from_pretrained(args.model_name)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            
            # Ensure PAD token is properly set
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
                
            chat_loop(model, tokenizer)
            sys.exit(0)
            
        # Check if the model should be trained or loaded.
        if args.train or args.resume or not os.path.exists(args.output_dir):
            model, tokenizer = train_model(
                args.csv, 
                model_name=args.model_name, 
                output_dir=args.output_dir, 
                context_window=args.context_window, 
                epochs=args.epochs,
                resume_training=args.resume,
                max_samples=args.max_samples,
                sample_frac=args.sample_frac,
                debug=args.debug
            )
        else:
            try:
                # Check if model files exist
                config_path = os.path.join(args.output_dir, "config.json") 
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Model config not found at {config_path}")
                    
                # Load the saved model and tokenizer.
                model = AutoModelForCausalLM.from_pretrained(args.output_dir)
                tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
                
                # Ensure PAD token is properly set
                if tokenizer.pad_token is None:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    model.resize_token_embeddings(len(tokenizer))
                    
                # Check if metadata exists and log it
                metadata_path = os.path.join(args.output_dir, "model_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    logger.info(f"Loaded model metadata: {metadata}")
                    
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                logger.info("Deleting corrupted model directory and retraining...")
                shutil.rmtree(args.output_dir)
                model, tokenizer = train_model(
                    args.csv, 
                    model_name=args.model_name,
                    output_dir=args.output_dir, 
                    context_window=args.context_window, 
                    epochs=args.epochs
                )

        # Start the interactive chat loop.
        chat_loop(model, tokenizer)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"An error occurred: {e}")
        print("Check the log file for details.")
