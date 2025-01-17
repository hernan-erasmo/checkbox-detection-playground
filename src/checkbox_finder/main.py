import cv2


def process_image(image_path, output_path, debug_mode: bool = False):
    print("process_image - Loading the image")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug_mode:
        cv2.imwrite("1-process-image-gray.png", gray)

    print("process_image - Preprocessing the image")
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(binary, 50, 150)
    if debug_mode:
        cv2.imwrite("2-process-image-binary.png", binary)
        cv2.imwrite("3-process-image-edges.png", edges)

    print("process_image - Finding contours")
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if debug_mode:
        contour_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)
        cv2.imwrite("4-process-image-contours.png", contour_image)

    print(f"process_image - Found {len(contours)} contours")
    for idx, contour in enumerate(contours):
        print(f"{idx+1}/{len(contours)} - Approximating to polygon")
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        print(f"{idx+1}/{len(contours)} - Filtering for rectangular shapes")
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # TODO: Fine-tune size/aspect ratio for checkboxes
            if 0.9 <= aspect_ratio <= 1.1 and 10 <= w <= 50:
                # Analyze the fill density inside the rectangle
                print(f"{idx+1}/{len(contours)} - Analyzing fill density of rectangle")
                roi = binary[y : y + h, x : x + w]
                fill_ratio = cv2.countNonZero(roi) / (w * h)

                print(f"{idx+1}/{len(contours)} - Classifying as filled or unfilled")
                color = (
                    (0, 255, 0) if fill_ratio > 0.3 else (0, 0, 255)
                )  # Green for filled, Red for unfilled
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                # TODO: Optimize fill ratio threshold

    # Save the output image
    print("process_image - Saving the output image")
    cv2.imwrite(output_path, image)


# Instructions for running the script
if __name__ == "__main__":
    input_path = "0-input.webp"  # Replace with your input file path
    output_path = "output_highlighted.png"
    process_image(input_path, output_path, True)
    print(f"Processed image saved to {output_path}")
