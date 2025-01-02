## FANTAstic SEquences and Where to Find Them: Faithful and Efficient API Call Generation through State-tracked Constrained Decoding and Reranking

This repository contains the source code (huggingface transformers version) for the EMNLP 2024 findings paper: [FANTAstic SEquences and Where to Find Them: Faithful and Efficient API Call Generation through State-tracked Constrained Decoding and Reranking](https://arxiv.org/abs/2407.13945)


## Rquirements
[fast-soft-sort](https://github.com/google-research/fast-soft-sort/tree/master)

accelerate==0.21.0  
datasets==2.12.0  
en-core-web-md @ https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.6.0/en_core_web_md-3.6.0-py3-none-any.whl  
nltk==3.8.1  
numpy==1.24.4  
pandas==2.0.3  
scikit-learn==1.3.0  
scipy==1.10.1  
spacy==3.6.1  
torch==2.0.1+cu117  
tqdm==4.66.1  
transformers==4.33.2  


## Prepare Constraints 

### Usage 

```
python prepare_constraints.py [-h] [--config_file CONFIG_FILE]
                              [--output_dir OUTPUT_DIR] [--model MODEL]
                              [--api_data API_DATA]
```

### Optional arguments 

```
   -h, --help            show this help message and exit
  --config_file CONFIG_FILE
                        Optional use of config file for passing the arguments
  --output_dir OUTPUT_DIR
                        The output directory where the processed constraints
                        will be written
  --model MODEL         Huggingface model name / Recorded model name / local
                        path to model directory
  --api_data API_DATA   Recorded api data name / Path to API data
```

## State-tracked Constrained Decoding 

### Usage 

```
python constrained_apibank.py [-h] --model_path MODEL_PATH --dataset_path
                              DATASET_PATH --output_file OUTPUT_FILE
                              --split_type SPLIT_TYPE
                              [--gpu_device GPU_DEVICE]
                              [--config_file CONFIG_FILE]
                              [--constraints_path CONSTRAINTS_PATH]
                              [--beam_size BEAM_SIZE] [--topkp] [--k K]
                              [--p P] [--max_new_tokens MAX_NEW_TOKENS]
```

### Optional arguments 

```
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
  --dataset_path DATASET_PATH
  --output_file OUTPUT_FILE
  --split_type SPLIT_TYPE
  --gpu_device GPU_DEVICE
                        GPU device id
  --config_file CONFIG_FILE
                        (optional) loading arguments with config file
  --constraints_path CONSTRAINTS_PATH
                        Path to folder that hosts the constraints files
  --beam_size BEAM_SIZE
                        Size of the beam search
  --topkp               Use top-k or top-p sampling
  --k K                 Value of k for top-k sampling, use 1 for greedy search
  --p P                 Value of p for top-p sampling, default to 0.9, will be
                        overrided if k is set
  --max_new_tokens MAX_NEW_TOKENS
                        Maximum amount of tokens to generation, default to 200
```

## Prepare Reranker Training Data 

### Usage 

```
python prepare_reranker_training_data.py
```

### Optional arguments 

Follow the prompt. 

## Train Reranking Model 

### Usage 

```
python train_reranker.py [-h] [--config_file CONFIG_FILE]
                         [--output_dir OUTPUT_DIR] [--gpu_device GPU_DEVICE]
                         [--train_set TRAIN_SET] [--test_set TEST_SET]
                         [--max_input_length MAX_INPUT_LENGTH] [--do_train]
                         [--seed SEED] [--num_epochs NUM_EPOCHS]
                         [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
                         [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
                         [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                         [--learning_rate LEARNING_RATE]
                         [--scheduler_type SCHEDULER_TYPE]
                         [--warmup_steps WARMUP_STEPS]
```

### Optional arguments 

```
  -h, --help            show this help message and exit
  --config_file CONFIG_FILE
                        Optional use of config file for passing the arguments
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written
  --gpu_device GPU_DEVICE
                        GPU device id
  --train_set TRAIN_SET
                        Path to the file of training data
  --test_set TEST_SET   Path to the file of testing data
  --max_input_length MAX_INPUT_LENGTH
                        Maximum input length after tokenization
  --do_train            Whether to run training.
  --seed SEED           Random seed
  --num_epochs NUM_EPOCHS
                        Total number of training epochs to perform.
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
                        Batch size per GPU/CPU for training
  --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE
                        Batch size per GPU/CPU for evaluation
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass; effective training
                        batch size equals to per_gpu_train_batch_size *
                        gradient_accumulation_steps
  --learning_rate LEARNING_RATE
                        The initial learning rate of AdamW for the model
  --scheduler_type SCHEDULER_TYPE
                        Learning rate scheduler type (linear/cosine/cosine_wit
                        h_restarts/polynomial/constant/constant_with_warmup)
  --warmup_steps WARMUP_STEPS
```

## Sample DSTC8 and API Bank Data 

### DSTC8 

Input: 

```
You are a helpful AI Assistant having a dialog with a Human user. You fulfill the user request by calling APIs and provide informative responses. Use the following format.
Human: the input utterance of the user. 
Action: take an action to fulfill the user's request by calling APIs. Stop generation with [EOM]
Assistant: an informative response to the user. 

You have access to the user's context and the following APIs.
Services_1.BookAppointmentDoc("doctor_name" : Required, "appointment_time" : Required, "appointment_date" : Required) Book an appointment with a specific doctor for the given date and time. . "doctor_name" is a string type."doctor_name" refers to name of the doctor or the medical practice, "doctor_name" is a non-categorical value, the possible values for "doctor_name" include [], and the argument "doctor_name" is required. . "appointment_time" is a string type."appointment_time" refers to time for the appointment with the doctor, "appointment_time" is a non-categorical value, the possible values for "appointment_time" include [], and the argument "appointment_time" is required. . "appointment_date" is a string type."appointment_date" refers to date for scheduling the appointment with the doctor, "appointment_date" is a non-categorical value, the possible values for "appointment_date" include [], and the argument "appointment_date" is required. . 

RentalCars_1.ReserveCar("pickup_location" : Required, "pickup_date" : Required, "pickup_time" : Required, "dropoff_date" : Required, "type" : Required) Reserve car rental for given dates and location. . "pickup_location" is a string type."pickup_location" refers to location of rental car pickup, "pickup_location" is a non-categorical value, the possible values for "pickup_location" include [], and the argument "pickup_location" is required. . "pickup_date" is a string type."pickup_date" refers to date of rental car pickup, "pickup_date" is a non-categorical value, the possible values for "pickup_date" include [], and the argument "pickup_date" is required. . "pickup_time" is a string type."pickup_time" refers to time of rental car pickup, "pickup_time" is a non-categorical value, the possible values for "pickup_time" include [], and the argument "pickup_time" is required. . "dropoff_date" is a string type."dropoff_date" refers to date of rental car drop-off, "dropoff_date" is a non-categorical value, the possible values for "dropoff_date" include [], and the argument "dropoff_date" is required. . "type" is a string type."type" refers to category to which rental car belongs, "type" is a categorical value, the possible values for "type" include ["Compact", "Standard", "Full-size"], and the argument "type" is required. . 


Example workflow similar to current utterance:
Human: Hello. I need a to use rental car from Saturday this week until Sunday this week.
Assistant: At what time of day and location?.
Human: I want to pick it up at 6:30 in the evening in SFO.
Assistant: There's a standard Chevrolet Cruze you can pick up from SFO International Airport on March 2nd that I highly recommend..
Human: What will it cost me?
Assistant: A total of $35..
Human: Sounds good to me.
Assistant: Want to reserve this car?.
Human: I want to do that, yes.
Assistant: Okay. You're going to reserve a standard car from SFO International Airport at 6:30 pm on March 2nd, and will give it back on March 3rd?.
Human: That is all correct.
Assistant: I reserved your car..
Human: Okay, great! I also want to watch a movie while I'm there.
Assistant: While in San Francisco..
Human: Yes, that's right.
Action: Movies_1.FindMovies(location="San Francisco")[EOM]

Human: I need to find a doctor.
Assistant: In what city?.
Human: Look for one in Pleasant Hill.
Assistant: What type of doctor do you need?.
Human: I need to find a General Practitioner.
Assistant: I found 5 possible results. Dr. Chika Akera, MD is a General Practitioner located in Pleasant Hill..
Human: That will work.
Assistant: Would you like to make an appointment?.
Human: No, maybe later.
Assistant: Anything else you need?.
Human: When am I free on the 12th?
Assistant: On March 12th you are available from 11:30 am until 6 pm..
Human: That's good. Can you make an appointment with the doctor on March 12th? Make it for half past 11 in the morning.
Assistant: Please confirm the following: You want an appointment with Dr. Chika Akera, MD. for March 12th at 11:30 am..
Human: Yes, that is correct. Can you give me their rating?
Action: Services_1.BookAppointmentDoc(appointment_time="11:30", doctor_name="Dr. Chika Akera, MD", appointment_date="2019-03-12")[EOM]


Context:

Human: I'm overdue for a few doctors visits. Can you help me find a gynecologist in San Jose?
Assistant: Sure, Anjali Tate, M.D. looks like a good Gynecologist in San Jose. How about that?.
Human: Yeah, that sounds good.
Assistant: Do you want to make an appointment?.
Human: Yes, can you do that for me?
Assistant: Sure, what time?.
Human: Make it for ten am.
Assistant: Ok, and what day?.
Human: Thursday next week would be ideal.
Assistant: Ok. To confirm, you want to make an appointment with Anjali Tate, M.D. on March 7th at 10 am. Is that right?.
Human: Yes, that's right.
Action:
```

Gold: 

```
Action: Services_1.BookAppointmentDoc(appointment_date="2019-03-07", appointment_time="10:00", doctor_name="Anjali Tate, M.D.")
```

### API Bank 

Input: 

```
Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the previous dialogue context.
The current year is 2023.
Input: 
User: User's utterence
AI: AI's response

Expected output:
API-Request: [ApiName(key1='value1', key2='value2', ...)]

API descriptions:
{"name": "QueryHealthData", "description": "This API queries the recorded health data in database of a given user and time span.", "input_parameters": {"user_id": {"type": "str", "description": "The user id of the given user. Cases are ignored."}, "start_time": {"type": "str", "description": "The start time of the time span. Format: %Y-%m-%d %H:%M:%S"}, "end_time": {"type": "str", "description": "The end time of the time span. Format: %Y-%m-%d %H:%M:%S"}}, "output_parameters": {"health_data": {"type": "list", "description": "The health data of the given user and time span."}}}
{"name": "CancelRegistration", "description": "This API cancels the registration of a patient given appointment ID.", "input_parameters": {"appointment_id": {"type": "str", "description": "The ID of appointment."}}, "output_parameters": {"status": {"type": "str", "description": "The status of cancellation."}}}
{"name": "ModifyRegistration", "description": "This API modifies the registration of a patient given appointment ID.", "input_parameters": {"appointment_id": {"type": "str", "description": "The ID of appointment."}, "new_appointment_date": {"type": "str", "description": "The new appointment date. Format: %Y-%m-%d."}, "new_appointment_doctor": {"type": "str", "description": "The new appointment doctor."}}, "output_parameters": {"status": {"type": "str", "description": "The status of modification."}}}

Input:
User: Can you please modify my appointment scheduled for March 25th with Dr. Kim to March 26th with Dr. Lee?
AI: Sure, I can help you with that. Please provide me with the appointment ID and the new appointment date and doctor's name.
User: The appointment ID is 34567890 and the new date is March 26th with Dr. Lee.
AI: Alright. I'll modify your appointment now.
Generate API Request:
```

Gold: 

```
API-Request: [ModifyRegistration(appointment_id='34567890', new_appointment_date='2023-03-26', new_appointment_doctor='Dr. Lee')]
```

## Adapt FANTASE to New API Formats 
In this repository, we provide the source code for the major components of FANTASE. Below are the steps required to adapt FANTASE to new API formats and constraints:  
1) Identify the structural tokens of the new format with the generation model's tokenizer  
2) Follow L89~116 of prepare_constraints.py to define the new structural tokens  
3) Follow the remaining sections of prepare_constraints.py to parse the api documents and construct the constrained token search trie (using nested dictionary to represent the trie)  
4) Based on constrained_apibank.py and constrained_dstc8.py, update structural tokens defined in L92+ and adjust the constrained decoding functions to reflect the new structural tokens and constraints  

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.