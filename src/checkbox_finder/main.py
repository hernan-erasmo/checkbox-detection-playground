import cv2


def process_image(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocessing
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(binary, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour to a polygon
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        # Filter for rectangular shapes
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # TODO: Fine-tune size/aspect ratio for checkboxes
            if 0.9 <= aspect_ratio <= 1.1 and 10 <= w <= 50:
                # Analyze the fill density inside the rectangle
                roi = binary[y : y + h, x : x + w]
                fill_ratio = cv2.countNonZero(roi) / (w * h)

                # Classify as filled or unfilled
                color = (
                    (0, 255, 0) if fill_ratio > 0.3 else (0, 0, 255)
                )  # Green for filled, Red for unfilled
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                # TODO: Optimize fill ratio threshold

    # Save the output image
    cv2.imwrite(output_path, image)


# Instructions for running the script
if __name__ == "__main__":
    input_path = "input.webp"  # Replace with your input file path
    output_path = "output_highlighted.png"
    process_image(input_path, output_path)
    print(f"Processed image saved to {output_path}")
