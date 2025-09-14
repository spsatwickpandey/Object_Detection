# Live Camera Object Detection with YOLOv8

A real-time object detection web application using YOLOv8 and Flask. This application provides live camera object detection with a modern, responsive UI.

## Features

- 🎥 **Live Camera Detection**: Real-time object detection using your webcam
- 🎯 **YOLOv8 Integration**: Powered by Ultralytics YOLOv8 for accurate object detection
- 📊 **Unique Object Tracking**: Tracks unique objects with timestamps and detection counts
- 🎨 **Modern UI**: Beautiful, responsive interface built with Tailwind CSS
- 📱 **Mobile Friendly**: Works on desktop and mobile devices
- ⚡ **Real-time Processing**: Fast inference with bounding box visualization

## Live Demo

The application is deployed on Render and can be accessed at: [Your Render URL]

## Technology Stack

- **Backend**: Flask (Python)
- **AI Model**: YOLOv8 (Ultralytics)
- **Frontend**: HTML5, JavaScript, Tailwind CSS
- **Computer Vision**: OpenCV
- **Deployment**: Render

## Local Development

### Prerequisites

- Python 3.10+
- Webcam or camera device

### Installation

1. Clone the repository:
```bash
git clone https://github.com/spsatwickpandey/Object_Detection.git
cd Object_Detection
```

2. Create a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Usage

1. **Start Camera**: Click the "Start Camera" button to begin live detection
2. **View Detections**: Objects will be highlighted with bounding boxes in real-time
3. **Track History**: View unique objects detected with timestamps in the history panel
4. **Stop Camera**: Click "Stop Camera" to end the detection session

## API Endpoints

- `GET /`: Main application interface
- `POST /upload`: Process camera frames for object detection

## Deployment

This application is configured for deployment on Render using the `render.yaml` configuration file.

### Render Deployment Steps

1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Use the following settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn wsgi:app`
   - **Python Version**: 3.10.0

## Project Structure

```
Object_Detection/
├── app.py                 # Main Flask application
├── wsgi.py               # WSGI entry point for deployment
├── requirements.txt      # Python dependencies
├── render.yaml          # Render deployment configuration
├── templates/
│   └── index.html       # Main HTML template
├── static/
│   └── favicon.ico      # Website favicon
└── README.md            # This file
```

## Features in Detail

### Live Detection
- Real-time camera feed processing
- Automatic object detection with confidence scores
- Bounding box visualization with labels
- Detection count indicator

### Unique Object History
- Tracks unique objects detected over time
- Shows first seen timestamp
- Displays detection count for each object
- Clear history functionality

### Responsive Design
- Mobile-first design approach
- Dark theme with modern UI elements
- Smooth animations and transitions
- Cross-browser compatibility

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for the YOLOv8 model
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Tailwind CSS](https://tailwindcss.com/) for the styling framework
- [Render](https://render.com/) for the deployment platform
