AI Caption Generator Project
Authors

Laiba Muneer (st125496@ait.asia)
Lakshika P. M. M. (st124872@ait.asia)
Aakash Kuragayala (st125050@ait.asia)Affiliation: Department of Information and Communication Technology, School of Engineering and Technology, Asian Institute of Technology

Overview
This project develops an AI-powered caption generator for social media posts, focusing on generating trendy, Gen Z-style captions for categories such as events, places, products, and experiences. Initially, the project attempted to scrape captions from online sources, but due to challenges with web scraping, the dataset was ultimately generated synthetically using multiple AI models, including GPT, Copilot, Claude, Gemini, Grok, DeepSeek, and others. The captions were then preprocessed, analyzed, and used to fine-tune a language model with Retrieval-Augmented Generation (RAG). The final application offers an interactive chatbot and a Streamlit-based web interface for generating captions.
The project evolved through several stages: from web scraping attempts to synthetic data generation, exploratory data analysis (EDA), model fine-tuning, and deployment as an interactive application.

Project Journey
Initial Approach: Web Scraping
What I Tried First:

The initial goal was to collect a large dataset of promotional captions to train a caption generation model.
I attempted to scrape websites like latestly.com, ndtv.com, timesofindia.indiatimes.com, and others for categories such as places, events, products, marketing, entertainment, and lifestyle.
I used BeautifulSoup to scrape captions from headings and paragraphs that matched promotional keywords (e.g., "visit", "join", "celebrate").
I also tried scraping Instagram (InstagramScrapper class) and Twitter (TwitterScrapper class) using Selenium to handle dynamic content.
Additionally, I explored the VQA v2.0 dataset (vqa_text_data.csv) to repurpose question-answer pairs as captions.

Challenges:

Many websites required JavaScript rendering, which BeautifulSoup couldn’t handle effectively.
Anti-scraping measures blocked requests, leading to incomplete data collection.
Scraping Instagram and Twitter was slow, and the data was noisy with duplicates and irrelevant content.
The VQA dataset, while useful, wasn’t directly aligned with social media caption styles.
Overall, web scraping couldn’t provide the volume or quality of data needed for the project.

What I Did:

Recognizing the limitations of web scraping, I pivoted to synthetic data generation using multiple AI models to create a high-quality, diverse dataset of captions.
I used models like GPT, Copilot, Claude, Gemini, Grok, and DeepSeek to generate captions for categories like events, places, products, and experiences.
Prompts were designed to encourage trendy, Gen Z-style captions (e.g., "Generate a trendy social media caption for a new tech product launch").
The generated captions were collected into captions.txt and converted to captions.csv for further processing.

Data Generation with AI Models
How I Generated the Data:

I leveraged multiple AI models to generate captions:
GPT: Used for general-purpose caption generation with prompts tailored to different categories.
Copilot: Assisted in generating creative and code-related captions.
Claude: Provided conversational and engaging captions with a focus on storytelling.
Gemini: Generated captions despite rate limits (60/minute), focusing on promotional styles.
Grok: Created humorous and unique captions with a fresh perspective.
DeepSeek: Focused on concise and trendy captions suitable for social media.


Each model was prompted with category-specific instructions to ensure diversity (e.g., "Generate a Gen Z-style caption for a music festival event").
The generated captions were filtered for length (≤280 characters) and relevance (e.g., containing keywords like "join", "explore", "celebrate").
I also added 20-25 unique hashtags per category to make the captions social media-ready (e.g., #WanderlustWavy for places).

Outcome:

This approach resulted in a robust dataset of captions (captions.csv) that was more diverse, trendy, and aligned with social media needs compared to scraped data.
The dataset was free from the noise and inconsistencies of web scraping, though it required careful curation to avoid model-specific biases.

Data Analysis and Preprocessing
What I Tried:

I performed EDA on the generated captions (captions.csv) to understand their characteristics.
Visualizations included bar charts for category distribution, histograms for character/word length, and word clouds for common words.
I used NLTK for tokenization and stopword removal to analyze word frequencies.

Challenges:

Some generated captions were repetitive due to model biases (e.g., overusing certain phrases).
Balancing the dataset across categories required manual curation.
Ensuring the captions maintained a Gen Z vibe across all models was challenging.

What I Did:

I removed duplicates and filtered out irrelevant captions during preprocessing.
I added features like word count, character length, and hashtag presence to the dataset.
I categorized captions into types (e.g., lifestyle, places, events) and ensured each caption had trendy hashtags.
The cleaned dataset was saved as cleaned_promotional_captions.csv.

Model Training and Fine-Tuning
What I Tried:

I fine-tuned a distilgpt2 model on the generated captions (captions.txt) to create a caption generator.
I integrated Retrieval-Augmented Generation (RAG) using sentence-transformers and faiss to retrieve relevant context for generation.
I used the Hugging Face transformers library for training and generation.

Challenges:

Initial fine-tuning produced generic captions that lacked the trendy, Gen Z vibe.
RAG integration required careful tuning to ensure retrieved captions were relevant.
Training logs were not always accessible, making performance analysis difficult.

What I Did:

I fine-tuned the distilgpt2 model in two stages: first for 10 epochs (caption_model_trained/model.pth) and then for 2 additional epochs (caption_model_finetuned/model.pth) with a lower learning rate to improve caption quality.
I used RAG with FAISS to retrieve relevant captions as context, enhancing the relevance of generated captions.
I incorporated Chain-of-Thought (CoT) prompting to guide the model in generating trendy captions with hashtags.
I analyzed training metrics (e.g., loss decrease) by parsing logs when available.

Final Implementation: Interactive Chatbot and Streamlit UI
What I Ended Up Doing:

I built an interactive chatbot that takes user prompts (e.g., "I had a Thailand trip I enjoyed a lot") and generates Gen Z-style captions using the fine-tuned distilgpt2 model with RAG and CoT.
I developed a Streamlit-based web interface (st.title("AI Caption Generator")) for users to input prompts and generate captions with a modern UI.
The final application uses the fine-tuned model (./fine_tuned_caption_model/model.pth), retrieves context with FAISS, and adds trendy hashtags.
I saved all project files in a ZIP (colab_all_files.zip) for easy sharing.


Project Structure

captions.txt: Synthetically generated captions from multiple AI models.
captions.csv: Converted captions in CSV format.
cleaned_promotional_captions.csv: Cleaned captions with hashtags.
train_captions.txt: Training file for fine-tuning the model.
fine_tuned_caption_model/model.pth: Fine-tuned DistilGPT-2 model weights.
logs/, logs_finetune/: Training logs (if available).
category_distribution.png, text_length_distribution.png, word_cloud.png: EDA visualizations.


Requirements
To run this project, install the following dependencies:
pip install pandas torch transformers sentence-transformers faiss-cpu langchain langchain-huggingface
pip install streamlit matplotlib seaborn wordcloud nltk

Additionally:

Download NLTK data: nltk.download('punkt') and nltk.download('stopwords').


How to Run
1. Prepare Data
The captions dataset (captions.txt) is already generated using multiple AI models. If you need to generate more captions, you can use the respective APIs of GPT, Claude, Gemini, Grok, DeepSeek, etc., with appropriate prompts.
2. Preprocess and Analyze Data
Run the preprocessing and EDA script to clean and analyze the captions:
python <preprocessing_script>.py


This will clean the data, add hashtags, and generate visualizations (category_distribution.png, word_cloud.png).

3. Fine-Tune the Model
Run the fine-tuning script to train the distilgpt2 model:
python <fine_tune_script>.py


This will save the fine-tuned model to ./fine_tuned_caption_model/model.pth.

4. Run the Interactive Chatbot
Run the chatbot script to generate captions interactively:
python <chatbot_script>.py


Enter prompts like "I had a Thailand trip I enjoyed a lot" to generate captions.

5. Run the Streamlit Web App
Run the Streamlit app for a web-based interface:
streamlit run <streamlit_script>.py


Open the provided URL in your browser, enter a prompt, and generate captions.


Example Usage
Chatbot
Prompt: I had a Thailand trip I enjoyed a lot
Generated Caption: Thailand trip hittin’ different! 🌴 Vibes on fleek #WanderlustWavy #LitLocations

Streamlit App

Input: "Launching a new tech product 🚀"
Output: "Tech game strong with this launch! 🚀 Cop it now #CopThis #TechSzn"


Future Improvements

Dynamic Trends: Integrate a real-time trends API (e.g., Twitter Trends) to add up-to-date hashtags.
Improved RAG: Use a larger and more diverse context dataset for better retrieval.
Multimodal Support: Incorporate image inputs to generate captions based on images.
Deployment: Host the Streamlit app on a cloud platform for public access.


License
This project is licensed under the MIT License. Feel free to use and modify it as needed.
