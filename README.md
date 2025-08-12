# YouTube RAG (Retrieval-Augmented Generation)

A powerful application that allows you to interact with YouTube videos using AI-powered question answering. This project uses Retrieval-Augmented Generation (RAG) to transcribe YouTube videos and provide intelligent responses to questions about the video content.

## 🚀 Features

- **YouTube Video Transcription**: Automatically extracts transcripts from YouTube videos
- **AI-Powered Q&A**: Ask questions about video content and get intelligent responses
- **Streamlit Web Interface**: User-friendly web application for easy interaction
- **Vector Database**: Uses FAISS for efficient document retrieval
- **Advanced Text Processing**: Intelligent text chunking and embedding generation
- **Groq LLM Integration**: Powered by Llama 3.3 70B model for high-quality responses

## 🛠️ Technology Stack

- **Python 3.8+**
- **Streamlit** - Web application framework
- **LangChain** - LLM orchestration and RAG implementation
- **FAISS** - Vector similarity search
- **Hugging Face** - Text embeddings and transformers
- **Groq** - High-performance LLM API
- **YouTube Transcript API** - Video transcription

## 📋 Prerequisites

Before running this application, make sure you have:

1. **Python 3.8 or higher** installed
2. **Groq API Key** - Get your free API key from [Groq](https://console.groq.com/)
3. **Internet connection** for YouTube video access and model downloads

## 🚀 Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd youtube_RAG
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   If requirements.txt doesn't exist, install manually:
   ```bash
   pip install streamlit langchain langchain-groq langchain-community faiss-cpu sentence-transformers youtube-transcript-api
   ```

3. **Set up your Groq API key**:
   - Get your API key from [Groq Console](https://console.groq.com/)
   - Update the API key in `ulits/helperfunction.py` (line 22)

## 🎯 Usage

### Web Application (Recommended)

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the provided URL (usually `http://localhost:8501`)

3. **Enter a YouTube video URL** in the text input field

4. **Get instant answers** about the video content

### Jupyter Notebook

1. **Open the notebook**:
   ```bash
   jupyter notebook yt_rag.ipynb
   ```

2. **Run the cells** to test the functionality with your own YouTube URLs

## 🔧 How It Works

1. **Video Processing**: The application takes a YouTube URL and extracts the video transcript
2. **Text Chunking**: Long transcripts are split into manageable chunks (2000 characters with 100 character overlap)
3. **Embedding Generation**: Text chunks are converted to vector embeddings using Hugging Face models
4. **Vector Storage**: Embeddings are stored in a FAISS vector database for fast similarity search
5. **Query Processing**: When you ask a question, the system finds the most relevant text chunks
6. **AI Response**: The Groq LLM generates a comprehensive answer based on the retrieved context

## 📁 Project Structure

```
youtube_RAG/
├── app.py                 # Streamlit web application
├── yt_rag.ipynb         # Jupyter notebook for testing
├── ulits/
│   └── helperfunction.py # Core RAG functionality
└── README.md             # This file
```

## 🔑 Configuration

### API Keys
- **Groq API Key**: Required for LLM responses
- Update the API key in `ulits/helperfunction.py`:
  ```python
  chat = ChatGroq(api_key = "YOUR_API_KEY_HERE", model = "llama-3.3-70b-versatile")
  ```

### Model Parameters
- **Chunk Size**: 2000 characters (configurable in `helperfunction.py`)
- **Chunk Overlap**: 100 characters
- **Similarity Search**: Top 4 most relevant chunks (configurable)
- **Temperature**: 0 (deterministic responses)

## 🎨 Customization

### Adding New Models
You can easily switch to different embedding models by modifying the embeddings initialization in `app.py`:

```python
# Example: Using a different Hugging Face model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### Modifying Chunk Parameters
Adjust text chunking in `helperfunction.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Smaller chunks for more granular search
    chunk_overlap=200  # More overlap for better context
)
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your Groq API key is valid and properly set
2. **Video Not Available**: Some videos may have disabled transcripts or be private
3. **Model Download Issues**: First run may take time to download embedding models
4. **Memory Issues**: Large videos may require more RAM for processing

### Solutions

- **Check API Key**: Verify your Groq API key is correct and has sufficient credits
- **Video Accessibility**: Ensure the YouTube video is public and has available transcripts
- **Clear Cache**: Remove downloaded models from `~/.cache/huggingface/` if needed
- **Reduce Chunk Size**: Lower chunk size for memory-constrained environments

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 🙏 Acknowledgments

- **LangChain** team for the excellent RAG framework
- **Groq** for providing high-performance LLM access
- **Hugging Face** for open-source embedding models
- **YouTube Transcript API** for video transcription capabilities

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the code comments for implementation details
3. Open an issue in the repository
4. Check the LangChain and Groq documentation for API-specific questions

---

**Happy Video Learning! 🎥✨**
