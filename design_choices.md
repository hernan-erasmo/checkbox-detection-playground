## Design decisions

### Language and libraries
Decided to do this in Python since it's the language I'm more experienced with. Quickly realized that OpenCV was one of the best choices if the idea was to not use a boxed solution, and *have low-level control over how the algorithm was implemented*.

### Performance
Before industrializing this app it would be a great idea to create a similar prototype in a lower level language (like C++, or Rust, which I'm more experienced with). Lots of math operations involved in this project and I believe there's a lot of room to improve on that front. Even though a single image is processed in under 1 second, it can quickly add up.

Another thing to consider is the possibility of applying parallelism and concurrency (probably more the former than the later, since this task is more CPU bound than I/O bound) to the operations that allow for it. In a pipeline that's supposed to handle millions of images, running multiple instances of the app (or the app on multiple cores) would be a must.

### Missing checkboxes
While the algorithm works, it still can't detect the following checkboxes
- *Utilities -> Electricity -> Public* checkbox
- *Zoning Compliance -> No Zoning* checkbox

For the first one, I'm 99,9% sure the problem is the contour of the checkbox is not closed. I experimented with a couple of alternatives to mitigate this problem, like using the `cv2.morphologyEx()` function, but wasn't able to adjust the inputs properly to get the rest of the algorithm to detect the checkbox.

For the second one, I think the problem is the shape of the contour is distorted by the large handwritten line that goes from bottom-left to top-right. But, same as above, I wasn't able to create a programmatic way of filtering those out properly.

For these two in particular, and for any other one-off situation that might present on analyzing millions of forms, I thought about a couple of approaches:

#### Training a model

Training a model to detect irregular checkboxes would involve collecting a labeled dataset, augmenting it with transformations to mimic edge cases, and training a computer vision model such as YOLOv5 or Faster R-CNN. Tools like PyTorch could be used for the model training itself, while OpenCV would help with preprocessing images.

Data annotation tools like LabelImg would be required for labeling tasks. After training, the model should be validated with metrics like precision and recall, and integrated into the pipeline with a confidence threshold for edge cases. Good thing about this approach is that it's scalable and would continually improve by retraining with newly encountered edge cases.

#### Use Amazon's mTurk

I first heard about mTurk about 10 years ago, and was surprised to see it was still around in this LLM and AI filled ecosystem these days. Amazon MTurk offers a quick and scalable way to handle challenging checkboxes by leveraging human judgment. It’s particularly useful for tasks where automation struggles, such as interpreting irregularly drawn forms. By routing these cases to MTurk, the complex model refinement can be bypassed and accurate results can be obtained without significant upfront effort.

It can also be scaled dynamically based on workload. Although easy to set up and flexible, it makes an effective short-term solution only. Especially when the volume of difficult cases is high or varies unpredictably.

### Which one would be best?

Lot's of unknowns here. A significant engineering effort could be oriented towards training a model just to have that work wasted by a change in the form structure a couple of months after deploying it. On the other hand, although mTurk might offer prices as low as $0.01 per image, getting reliable and quick results from mTurk workers might prove to be many times more expensive than that. Also, images should be cropped in a way that prevents leaking sensitive information from the forms, so it's also not an alternative without risks.

For all these reasons expressed above, and without knowing more details, I personally would go with mTurk.

## TODOs and improvements in no particular order

1. Make the code generate a JSON file where keys are checkbox labels and values are whether they're checked or not. I think this would be super helpful and a natural second step in this processing pipeline. After all, the idea to ingest data automatically by detecting the checkboxes, and not to have a human being look at the processed image.
2. Convert the app to a library for better versioning and ease of integration into a larger codebase.
3. Thorough testing
4. Profiling, to detect hotspots and help decide which code could be a candidate for being re-written in a lower level language.
5. Better understanding of the way image formats might've affected the detection algorithm
