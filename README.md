# Checkbox Detection Tool

A Python script that detects and categorizes checkboxes in form images, highlighting checked and unchecked boxes.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd checkbox-detection-playground
```

2. Create and activate virtual environment:
```bash
python -m venv .
source ./bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic usage

```bash
python src/checkbox_finder/main.py -i path/to/input/image.jpg
```

### Options
- `-i, --input`: Path to input image (required)
- `-o, --output`: Path for output image (optional)
- `-d, --debug`: Enable debug mode (optional)

### Examples
Process image with default output name:
```bash
python src/checkbox_finder/main.py -i 0-input.webp
```

Specify custom output path
```bash
python src/checkbox_finder/main.py -i 0-input.webp -o 0-input-processed.png
```

Enable debug mode
```bash
python src/checkbox_finder/main.py -i 0-input.webp -d
```

### Debug mode

When debug mode is enabled (`-d`), the script generates additional output files:

- `1a-sharpened.png`: Sharpened grayscale image
- `1b-adaptive-thresh.png`: Thresholded binary image
- `2a-thresh-input.png`: Input for contour detection
- `2b-contours-filtered.png`: Detected checkbox contours
- `3-categorized-checkboxes.png`: Final categorization results
