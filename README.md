# Fashion Recommendation System

**AI-Powered Personalized Fashion Intelligence Platform**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18.3.1-61DAFB.svg)](https://reactjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-EE4C2C.svg)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-000000.svg)](https://flask.palletsprojects.com)

A comprehensive AI-powered fashion recommendation system that analyzes facial features, age, and gender to provide personalized fashion suggestions. The system combines deep learning, computer vision, and web scraping to deliver tailored fashion recommendations. ## Project Demo

[![Watch the video](https://img.youtube.com/vi/LyuEf2Xhhhg/0.jpg)](https://www.youtube.com/watch?v=LyuEf2Xhhhg)


## 📋 Table of Contents

- [🌟 Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [🎯 How It Works](#-how-it-works)
- [🛠️ API Endpoints](#️-api-endpoints)
- [🧪 Testing](#-testing)
- [🔧 Configuration](#-configuration)
- [📊 Technical Details](#-technical-details)
- [🤝 Contributing](#-contributing)
- [🙏 Acknowledgments](#-acknowledgments)
- [📞 Support](#-support)

## 🌟 Features

- **AI-Powered Analysis**: Deep learning models for age and gender detection using PyTorch and EfficientNet-B4
- **Facial Feature Extraction**: Analyzes 16+ facial attributes for personalized recommendations
- **Real-time Camera Capture**: OpenCV integration for live photo capture
- **LLM Fashion Intelligence**: Advanced language models generate creative, themed fashion recommendations
- **Web Scraping**: Automatically finds real fashion products from e-commerce platforms
- **Modern UI**: Beautiful React frontend with Tailwind CSS and shadcn/ui components
- **Interactive Experience**: User-controlled workflow with manual navigation between steps
- **Personalized Recommendations**: Tailored fashion suggestions based on facial features, body type, and occasion
- **Theme-Aware Styling**: Supports themed occasions (anime, sports, cultural events, etc.)
- **Real-time Processing**: Instant analysis and recommendation generation

## 🏗️ Architecture

### Backend (Flask API)
- **Flask REST API** with CORS support
- **PyTorch Models** for age/gender classification and facial feature extraction
- **LLM Integration** for generating fashion recommendations using Hugging Face
- **Web Scraping Engine** for finding real fashion products
- **OpenCV** for image processing and camera integration

### Frontend (React + Vite)
- **React 18** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **shadcn/ui** component library
- **React Router** for navigation

## 📁 Project Structure

```
Fashion-Recommendation-system/
├── backend/                          # Flask API server
│   ├── api.py                       # Main Flask application
│   ├── predict.py                   # Age/gender prediction logic
│   ├── ml.py                        # Machine learning utilities
│   ├── llm.py                       # LLM integration for recommendations
│   ├── web_scrapping.py             # Web scraping functionality
│   ├── celeb_face.py                # Celebrity face analysis
│   ├── main.py                      # Model training script
│   ├── manual_test.py               # Manual testing script
│   ├── extract_info.py              # Information extraction utilities
│   ├── quick_fix.py                 # System repair script
│   ├── gpu_check.py                 # GPU availability checker
│   ├── requirements.txt             # Python dependencies
│   ├── .env                         # Environment variables
│   ├── age_gender_model.pth         # Trained age/gender model
│   ├── celeba_imbalance_aware_classifier.pth  # Facial feature model
│   ├── age_scaler.pkl               # Age prediction scaler
│   ├── facial_features.json         # Extracted facial features
│   ├── llm_recommendations.json     # LLM-generated recommendations
│   ├── multi_scraped_output.json    # Scraped fashion products
│   ├── list_attr_celeba.csv         # CelebA attributes dataset
│   ├── list_bbox_celeba.csv         # CelebA bounding boxes
│   ├── list_eval_partition.csv      # CelebA evaluation partitions
│   ├── list_landmarks_align_celeba.csv # CelebA facial landmarks
│   ├── img_align_celeba/            # CelebA aligned images dataset
│   ├── utkface_aligned_cropped/     # UTKFace dataset
│   ├── uploads/                     # Uploaded images storage
│   ├── venv/                        # Python virtual environment
│   └── __pycache__/                 # Python compiled files
├── face-insight-studio/             # React frontend application
│   ├── src/
│   │   ├── components/              # Reusable UI components
│   │   ├── pages/                   # Application pages
│   │   ├── hooks/                   # Custom React hooks
│   │   ├── lib/                     # Utility functions
│   │   ├── assets/                  # Static assets
│   │   ├── App.tsx                  # Main App component
│   │   ├── App.css                  # App-specific styles
│   │   ├── main.tsx                 # Application entry point
│   │   ├── index.css                # Global styles
│   │   └── vite-env.d.ts            # Vite environment types
│   ├── public/                      # Public static files
│   ├── node_modules/                # Node.js dependencies
│   ├── package.json                 # Node.js dependencies
│   ├── package-lock.json            # Dependency lock file
│   ├── bun.lockb                    # Bun lock file
│   ├── vite.config.ts               # Vite configuration
│   ├── tailwind.config.ts           # Tailwind CSS configuration
│   ├── postcss.config.js            # PostCSS configuration
│   ├── components.json              # shadcn/ui components config
│   ├── eslint.config.js             # ESLint configuration
│   ├── tsconfig.json                # TypeScript configuration
│   ├── tsconfig.app.json            # App-specific TypeScript config
│   ├── tsconfig.node.json           # Node-specific TypeScript config
│   ├── index.html                   # HTML entry point
│   ├── .gitignore                   # Git ignore rules
│   └── README.md                    # Frontend documentation
├── .git/                            # Git repository data
├── .gitattributes                   # Git attributes configuration
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 18+** with npm
- **Git**
- **Webcam** (optional, for camera capture feature)

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/MOHILMANDAPE15/Fashion-Recommendation-system.git
   cd Fashion-Recommendation-system
   ```

2. **Set up Python environment**
   ```bash
   cd backend
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the Flask server**
   ```bash
   python api.py
   ```
   
   The backend will be available at `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd face-insight-studio
   ```

2. **Install Node.js dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```
   
   The frontend will be available at `http://localhost:5173`

## 🎯 How It Works

### 1. Image Upload/Capture
- Users can upload an image or capture a photo using their webcam
- Images are processed and sent to the backend API

### 2. AI Analysis Pipeline
- **Age & Gender Detection**: Deep learning models analyze the uploaded image
- **Facial Feature Extraction**: 16+ facial attributes are extracted and analyzed
- **User Input Integration**: Height, weight, and occasion preferences are combined with AI analysis

### 3. Fashion Recommendation Generation
- **LLM Processing**: Advanced language models generate creative, themed fashion recommendations
- **Personalization**: Recommendations are tailored based on detected features and user preferences

### 4. Product Discovery
- **Web Scraping**: System automatically searches for real fashion products
- **Product Matching**: Finds items that match the generated recommendations
- **Data Enrichment**: Adds prices, ratings, reviews, and product images

### 5. Results Display
- **Interactive UI**: Clean, modern interface displays recommendations
- **Product Gallery**: Shows actual products with details and purchase links
- **User Control**: Manual navigation allows users to view results when ready

## 🛠️ API Endpoints

### Core Endpoints

- `POST /api/upload` - Upload image and get fashion recommendations
- `POST /api/camera_capture` - Process camera-captured image
- `GET /api/health` - Health check endpoint

### Request Format

```json
{
  "image": "base64_encoded_image_data",
  "height": "170",
  "weight": "65",
  "occasion": "casual"
}
```

### Response Format

```json
{
  "success": true,
  "recommendations": [
    {
      "brand": "Nike",
      "title": "Product Name",
      "price": "₹1,199",
      "rating": "4.2",
      "reviews": "1,234",
      "image_url": "https://...",
      "product_url": "https://..."
    }
  ],
  "captured_image": "base64_encoded_preview"
}
```

## 🧪 Testing

### Backend Testing
```bash
cd backend
python manual_test.py           # Manual testing with webcam
```

### Frontend Testing
```bash
cd face-insight-studio
npm run lint                    # ESLint checks
npm run build                   # Production build test
```

## 🔧 Configuration

### Environment Variables
The backend includes a `.env` file with the following configuration:
```env
HF_TOKEN="hugging face token /read"
```

### Dependencies
- **Backend**: All Python dependencies are listed in `backend/requirements.txt`
- **Frontend**: All Node.js dependencies are listed in `face-insight-studio/package.json`

Simply run the installation commands in the [Getting Started](#-getting-started) section.

### Model Files
The system requires pre-trained models:
- `age_gender_model.pth` - Age and gender classification (EfficientNet-B4 based)
- `celeba_imbalance_aware_classifier.pth` - Facial feature extraction (CelebA dataset)

## 📊 Technical Details

### Machine Learning Models
- **Age/Gender Classification**: Custom CNN trained on UTKFace dataset
- **Facial Feature Extraction**: CelebA-based classifier for 16+ attributes
- **Image Processing**: OpenCV for preprocessing and camera integration

### Data Flow
1. Image → Preprocessing → Feature Extraction
2. Features → LLM → Fashion Recommendations
3. Recommendations → Web Scraping → Product Data
4. Product Data → Frontend → User Display

### Model Evaluation
The multi-task CNN pipeline was evaluated on gender recognition, facial-attribute prediction, and age-group estimation.

| Task | Dataset | Test Accuracy |
|---|---|---:|
| Gender Recognition | UTKFace | 95.1% |
| Facial Attributes | CelebA | 91.9% |
| Age-Group Estimation | UTKFace | 80.3% |

Validation accuracy was tracked across 25 training epochs. The curves indicate the strongest performance for gender recognition and facial attributes, while age-group estimation remains the more challenging task.

### Performance Optimizations
- **Model Caching**: Pre-loaded models for faster inference
- **Async Processing**: Non-blocking API calls
- **Data Persistence**: JSON caching for recommendations and products

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **UTKFace Dataset** for age/gender classification training
- **CelebA Dataset** for facial feature extraction
- **OpenCV Community** for computer vision tools
- **React & Vite Teams** for excellent frontend tooling
- **shadcn/ui** for beautiful, accessible components

## 📞 Support

For support, email [neelshroff03@gmail.com  mohilmandpe33@gmail.com] or open an issue on GitHub.

---


