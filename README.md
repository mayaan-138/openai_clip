# 🏥 Med-Gemma Medical Image Analyzer

A Streamlit web application that uses Google's Med-Gemma AI model to analyze medical images (X-rays, CT scans, MRI) and provide detailed medical reports.

## 🚀 Features

- **Medical Image Analysis**: Upload X-ray, CT scan, or MRI images
- **AI-Powered Reports**: Get detailed medical analysis using Med-Gemma
- **Custom Prompts**: Choose from predefined analysis types or write custom prompts
- **Modern UI**: Beautiful, responsive web interface
- **Report Download**: Export analysis results as markdown files
- **GPU/CPU Support**: Automatically uses GPU if available

## 📋 Prerequisites

- Python 3.9 or higher
- Hugging Face account (for model access)
- Internet connection (for model download)

## 🛠️ Installation

1. **Clone or download this project**
   ```bash
   # If you have git installed
   git clone <repository-url>
   cd medgemma-assignment
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Hugging Face access**
   - Go to [huggingface.co](https://huggingface.co/join) and create an account
   - Go to [Med-Gemma model page](https://huggingface.co/google/med-gemma-2b)
   - Accept the model terms and conditions
   - (Optional) Create an access token for faster downloads)

## 🚀 Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

3. **First run setup**
   - On first run, the app will download the Med-Gemma model (~4GB)
   - This may take 5-10 minutes depending on your internet speed
   - The model will be cached for future use

## 📖 How to Use

1. **Upload an Image**
   - Click "Browse files" to upload a medical image
   - Supported formats: PNG, JPG, JPEG, BMP, TIFF
   - The image will be displayed in the left panel

2. **Choose Analysis Type**
   - Select from predefined analysis types in the sidebar:
     - General Analysis
     - X-Ray Analysis
     - CT Scan Analysis
     - MRI Analysis
     - Detailed Report
   - Or write your own custom prompt

3. **Analyze the Image**
   - Click the "🔍 Analyze Image" button
   - Wait for the AI to process the image (usually 10-30 seconds)
   - View the detailed medical report in the right panel

4. **Download Results**
   - Click "📥 Download Report" to save the analysis as a markdown file

## 🔧 Configuration

### Model Settings
- The app automatically uses the smallest Med-Gemma model (2B parameters) for faster inference
- GPU acceleration is automatically enabled if available
- Model is cached locally after first download

### Customization
You can modify the following in `model_utils.py`:
- Model parameters (temperature, max tokens, etc.)
- Different Med-Gemma model variants
- Custom prompts and analysis types

## ⚠️ Important Notes

### Disclaimer
- **This is a prototype for educational purposes only**
- **Not intended for clinical use**
- **Always consult healthcare professionals for medical decisions**
- The AI analysis should not replace professional medical diagnosis

### Model Limitations
- Med-Gemma is trained on medical data but may not be 100% accurate
- Results should be validated by medical professionals
- The model works best with clear, high-quality medical images

### System Requirements
- **Minimum**: 8GB RAM, CPU-only inference
- **Recommended**: 16GB+ RAM, GPU with 8GB+ VRAM
- **Storage**: ~4GB for model download

## 🐛 Troubleshooting

### Common Issues

1. **Model download fails**
   - Check your internet connection
   - Ensure you have accepted the model terms on Hugging Face
   - Try running with a Hugging Face access token

2. **Out of memory errors**
   - Close other applications to free up RAM
   - Try running on CPU only (slower but uses less memory)
   - Reduce image size before uploading

3. **Slow performance**
   - Ensure you're using a GPU if available
   - Try smaller images
   - Close other applications

### Getting Help
- Check the terminal output for error messages
- Ensure all dependencies are installed correctly
- Verify your Python version is 3.9 or higher

## 📁 Project Structure

```
medgemma-assignment/
├── app.py              # Main Streamlit application
├── model_utils.py      # Med-Gemma model handling
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🔄 Updates and Maintenance

- Keep your dependencies updated: `pip install -r requirements.txt --upgrade`
- Check for new Med-Gemma model versions on Hugging Face
- Monitor Streamlit and PyTorch updates for compatibility

## 📄 License

This project is for educational purposes. Please respect the Med-Gemma model license and terms of use.

---

**Built with ❤️ using Med-Gemma and Streamlit** 