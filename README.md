# AI Caption Generator Project

## Authors
- **Laiba Muneer** (st125496@ait.asia)  
- **Lakshika P. M. M.** (st124872@ait.asia)  
- **Aakash Kuragayala** (st125050@ait.asia)  

**Affiliation**: Department of Information and Communication Technology, School of Engineering and Technology, Asian Institute of Technology  

---

## Overview
This project develops an AI-powered caption generator for social media posts, focusing on generating trendy, Gen Z-style captions for categories such as events, places, products, and experiences. 

Initially, the project attempted to scrape captions from online sources, but due to challenges, the dataset was ultimately generated synthetically using multiple AI models, including GPT, Copilot, Claude, Gemini, Grok, DeepSeek, and others. The captions were then preprocessed, analyzed, and used to fine-tune a language model with Retrieval-Augmented Generation (RAG). 

The final application offers an interactive chatbot and a Streamlit-based web interface for generating captions.

---

## Project Journey

### Initial Approach: Web Scraping
**What I Tried First**:  
- Scraped websites like `latestly.com`, `ndtv.com`, and `timesofindia.indiatimes.com` for promotional captions.  
- Used **BeautifulSoup** for static scraping and **Selenium** for dynamic content (e.g., Instagram and Twitter).  
- Explored datasets like **VQA v2.0** to repurpose question-answer pairs as captions.

**Challenges**:  
- JavaScript-heavy websites were hard to scrape with BeautifulSoup.  
- Anti-scraping measures blocked requests.  
- Data from social platforms was noisy and inconsistent.

**Solution**: Pivoted to **synthetic data generation** using multiple AI models.

---

### Data Generation with AI Models
**How I Generated the Data**:  
- Used models like GPT, Claude, Copilot, Gemini, Grok, and DeepSeek with prompts tailored for categories (e.g., "Generate a trendy social media caption for a tech product launch").
- Filtered captions for length (≤280 characters) and relevance.  
- Added 20-25 unique hashtags per category (e.g., `#WanderlustWavy` for places).  

**Outcome**:  
- A diverse and trendy dataset stored in `captions.csv`.  
- Free from noise and inconsistencies compared to scraped data.

---

### Data Analysis and Preprocessing
**What I Did**:  
- Conducted **Exploratory Data Analysis (EDA)** on `captions.csv` using bar charts, histograms, and word clouds.  
- Removed duplicates and irrelevant captions.  
- Balanced categories manually and added trendy hashtags.  

**Outcome**: The cleaned dataset was saved as `cleaned_promotional_captions.csv`.

---

### Model Training and Fine-Tuning
**What I Tried**:  
1. Fine-tuned a `distilgpt2` model with captions (`captions.txt`).  
2. Integrated **Retrieval-Augmented Generation (RAG)** using `sentence-transformers` and `faiss`.  
3. Used **Chain-of-Thought (CoT)** prompting for trendy captions with hashtags.

**Challenges**:  
- Initial results lacked the Gen Z vibe.  
- RAG required careful tuning for relevance.  

**Solution**:  
- Fine-tuned for 10 epochs and 2 additional epochs with a lower learning rate.  
- Used RAG with FAISS to enhance relevance.  

---

### Final Implementation: Interactive Chatbot and Streamlit UI
**What I Did**:  
- Built an **interactive chatbot** to generate captions based on user prompts.  
- Developed a **Streamlit-based web interface** for a modern, user-friendly experience.  
- Saved all project files in `colab_all_files.zip` for easy sharing.

---

## Project Structure
- `captions.txt`: Synthetically generated captions.  
- `captions.csv`: Converted captions in CSV format.  
- `cleaned_promotional_captions.csv`: Cleaned captions with hashtags.  
- `train_captions.txt`: Training file for fine-tuning the model.  
- `fine_tuned_caption_model/model.pth`: Fine-tuned DistilGPT-2 model weights.  
- `logs/`, `logs_finetune/`: Training logs (if available).  
- `category_distribution.png`, `text_length_distribution.png`, `word_cloud.png`: EDA visualizations.

---

## Requirements
To run this project, install the following dependencies:

```bash
pip install pandas torch transformers sentence-transformers faiss-cpu langchain langchain-huggingface
pip install streamlit matplotlib seaborn wordcloud nltk
```

Additionally, download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## How to Run

### 1. Prepare Data
The captions dataset (`captions.txt`) is already generated. To generate more captions, use the APIs of GPT, Claude, Gemini, etc., with tailored prompts.

### 2. Preprocess and Analyze Data
Run the preprocessing and EDA script:
```bash
python <preprocessing_script>.py
```
This will clean the data and generate visualizations (`category_distribution.png`, `word_cloud.png`).

### 3. Fine-Tune the Model
Run the fine-tuning script:
```bash
python <fine_tune_script>.py
```
The fine-tuned model will be saved to `./fine_tuned_caption_model/model.pth`.

### 4. Run the Interactive Chatbot
Run the chatbot script:
```bash
python <chatbot_script>.py
```
Enter prompts like:
```
I had a Thailand trip I enjoyed a lot
```

### 5. Run the Streamlit Web App
Run the Streamlit app:
```bash
streamlit run <streamlit_script>.py
```
Open the provided URL in your browser and generate captions.

---

## Example Usage

### Chatbot
**Prompt**:  
```
I had a Thailand trip I enjoyed a lot
```

**Generated Caption**:  
```
Thailand trip hittin’ different! 🌴 Vibes on fleek #WanderlustWavy #LitLocations
```

### Streamlit App
**Input**:  
```
Launching a new tech product 🚀
```

**Output**:  
```
Tech game strong with this launch! 🚀 Cop it now #CopThis #TechSzn
```

---

## Future Improvements
- **Dynamic Trends**: Add real-time trends using APIs like Twitter Trends.  
- **Improved RAG**: Use a larger and more diverse context dataset.  
- **Multimodal Support**: Incorporate image input for caption generation.  
- **Deployment**: Host the Streamlit app on a public cloud platform.

---

## License
This project is licensed under the MIT License. Feel free to use and modify it as needed.
