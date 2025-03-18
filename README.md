# nlp_final_project

---

# AI-Powered Social Media Content Generator

**A Data-Driven Approach for Maximizing Engagement**

**Authors**:  
- Laiba Muneer (st125496@ait.asia)  
- Lakshika P. M. M. (st124872@ait.asia)  
- Aakash Kuragayala (st125050@ait.asia)  
**Affiliation**: Department of Information and Communication Technology, School of Engineering and Technology, Asian Institute of Technology

---

## Overview
Social media is a cornerstone of digital marketing, yet content creation remains time-consuming, inconsistent, and hard to scale with traditional methods. The **AI-Powered Social Media Content Generator** addresses these challenges by leveraging **Large Language Models (LLMs)**, **Retrieval-Augmented Generation (RAG)**, and **AI-driven analytics** to automate the generation of high-quality, platform-specific, and audience-optimized content. This system integrates real-time trend analysis, engagement prediction, and content personalization to maximize reach and interaction on platforms like Instagram, Twitter, and TikTok.

This project aims to revolutionize digital marketing by reducing manual effort, enhancing content relevance, and optimizing engagement through AI automation.

---

## Problem Statement
- **Time-Intensive Creation**: Manual content ideation and refinement struggle to keep up with social media’s pace.
- **Inconsistent Engagement**: Rapidly shifting trends and preferences make engagement unpredictable without data-driven insights.
- **Platform Complexity**: Each platform has unique formats and algorithms, complicating manual optimization.
- **Scalability**: High-quality content production at scale is unsustainable without automation.

---

## Motivation
The rapid evolution of digital marketing demands scalable, data-driven solutions. This project harnesses AI to:
- Automate content generation and optimization.
- Boost engagement through predictive analytics and trend adaptation.
- Empower businesses, influencers, and marketers with efficient, impactful social media strategies.

---

## System Architecture
The system is modular, integrating:
1. **Content Input & Trend Analysis**: Processes user inputs and fetches real-time trends (e.g., via X API).
2. **AI-Powered Content Generation**: Uses LLMs to create platform-tailored posts.
3. **Engagement Prediction**: Forecasts likes, shares, and comments using historical data.
4. **Optimization & Personalization**: Refines content based on audience demographics and platform metrics.
5. **User Interface**: Intuitive UI for content creation and performance tracking.

---

## Technology Stack
| **Component**             | **Tools & Frameworks**         |
|---------------------------|---------------------------------|
| LLM Framework            | OpenAI GPT, Groq, LangChain    |
| Retrieval Mechanism      | FAISS, ChromaDB, Pinecone      |
| Engagement Prediction    | TensorFlow, PyTorch            |
| SEO & Social Media Tools | Yoast SEO, Socialbakers API    |
| UI & Deployment          | Streamlit, FastAPI, AWS        |

---

## Methodology
1. **Data Processing**: Preprocesses user inputs and retrieves real-time trends using RAG.
2. **Content Generation**: Employs LLMs with Chain-of-Thought (CoT) prompting for coherent text.
3. **Optimization**: Enhances readability (e.g., Flesch-Kincaid) and SEO (e.g., keyword density).
4. **Model Training**: Fine-tunes LLMs and RAG models, optimizing hyperparameters for engagement.

---

## Datasets
| **Dataset**            | **Description**                          | **Purpose**                     |
|-----------------------|------------------------------------------|---------------------------------|
| Social Media Post     | 60 multilingual posts on trending topics | Multilingual text generation   |
| SynthFluencers        | Synthetic influencer profiles            | Personalization & influencer modeling |
| Flickr30k             | 30K images with captions                | Image-captioning for visual platforms |
| VQA                   | Images with question-answer pairs       | Context-aware visual content   |

**Preprocessing**: Text cleaning, tokenization, embedding generation (GPT/BERT), and multimodal feature extraction.

---

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/ai-social-media-generator.git
   cd ai-social-media-generator
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Example `requirements.txt`:
   ```
   tweepy
   transformers
   torch
   tensorflow
   streamlit
   fastapi
   faiss-cpu
   pinecone-client
   langchain
   ```

3. **Set Up API Keys**:
   - X API: Add your Bearer Token to `config.py` (e.g., `BEARER_TOKEN = "your_token"`).
   - Other APIs (e.g., Socialbakers): Configure as needed.

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

---

## Usage
1. **Input**: Specify post type, tone, keywords, and platform via the Streamlit UI.
2. **Generate**: AI creates optimized content based on trends and predictions.
3. **Review**: Evaluate engagement forecasts and refine output.
4. **Deploy**: Export posts or integrate with scheduling tools via FastAPI.

---

## Evaluation Metrics
- **BLEU Score**: Text fluency and coherence.
- **Engagement Accuracy**: Predicted vs. actual interaction metrics.
- **SEO Ranking**: Visibility and hashtag performance.
- **Readability Score**: Audience-tailored content quality.

---

## Expected Outcomes
- **Higher Engagement**: Optimized posts aligned with trends and algorithms.
- **Reduced Effort**: Automation of ideation and refinement.
- **Improved Visibility**: Enhanced SEO and platform ranking.

---

## Deployment
- **Web UI**: Streamlit for interactive use.
- **API**: FastAPI for integration with external tools.
- **Cloud**: AWS for scalability and real-time monitoring.

---

## Future Work
- Sentiment-based refinement.
- Multimodal integration for video content.
- Continuous learning for evolving trends.

---

## Contributing
Contributions are welcome! Please fork the repo, create a branch, and submit a pull request.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---

## Contact
For questions, reach out to:
- Laiba Muneer: st125496@ait.asia
- Lakshika P. M. M.: st124872@ait.asia
- Aakash Kuragayala: st125050@ait.asia

---

### Notes
- **GreenLens Tie-In**: If you want to merge this with GreenLens (eco-footprint focus), I can adjust it to highlight sustainability content generation—let me know!
- **Customization**: Replace `yourusername` in the git clone URL with your GitHub username.
- **Dependencies**: The `requirements.txt` is a sample—add specific versions or additional libraries (e.g., `chromadb`) as you finalize your stack.

Let me know if you’d like tweaks (e.g., adding a demo GIF, linking references, or aligning with GreenLens)!
