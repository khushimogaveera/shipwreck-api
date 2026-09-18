import cv2
import numpy as np
import torch


def predict_full_image(
    image,
    model,
    device,
    patch_size=512,
    stride=512,
    threshold=0.6
):
    """
    Predict a shipwreck segmentation mask for a full sonar image.

    Large images are resized for faster CPU inference.
    The final probability map is resized back to the
    original image dimensions.
    """

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    original_height, original_width = image.shape

    # --------------------------------------------------
    # Resize very large images for faster inference
    # --------------------------------------------------

    max_dimension = 3072

    scale = min(
        1.0,
        max_dimension / max(original_height, original_width)
    )

    if scale < 1.0:
        working_width = int(original_width * scale)
        working_height = int(original_height * scale)

        working_image = cv2.resize(
            image,
            (working_width, working_height),
            interpolation=cv2.INTER_AREA
        )
    else:
        working_image = image.copy()

    working_height, working_width = working_image.shape

    # Normalize
    working_image = (
        working_image.astype(np.float32) / 255.0
    )

    # --------------------------------------------------
    # Padding
    # --------------------------------------------------

    pad_h = max(
        0,
        patch_size - working_height
    )

    pad_w = max(
        0,
        patch_size - working_width
    )

    padded = np.pad(
        working_image,
        ((0, pad_h), (0, pad_w)),
        mode="reflect"
    )

    height, width = padded.shape

    # Add padding so the final patches cover the image
    extra_h = (
        (patch_size - (height - patch_size) % stride)
        % stride
    )

    extra_w = (
        (patch_size - (width - patch_size) % stride)
        % stride
    )

    if extra_h > 0 or extra_w > 0:
        padded = np.pad(
            padded,
            ((0, extra_h), (0, extra_w)),
            mode="reflect"
        )

    height, width = padded.shape

    probability_sum = np.zeros(
        (height, width),
        dtype=np.float32
    )

    count_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

    # --------------------------------------------------
    # Generate patches
    # --------------------------------------------------

    patches = []
    positions = []

    for y in range(
        0,
        height - patch_size + 1,
        stride
    ):
        for x in range(
            0,
            width - patch_size + 1,
            stride
        ):
            patches.append(
                padded[
                    y:y + patch_size,
                    x:x + patch_size
                ]
            )

            positions.append((y, x))

    # --------------------------------------------------
    # U-Net inference
    # --------------------------------------------------

    model.eval()

    with torch.no_grad():

        for start in range(
            0,
            len(patches),
            8
        ):

            batch_patches = patches[
                start:start + 8
            ]

            batch = np.stack(
                batch_patches
            )

            batch = torch.from_numpy(
                batch
            ).float()

            batch = batch.unsqueeze(1)

            batch = batch.to(device)

            outputs = model(batch)

            probabilities = torch.sigmoid(
                outputs
            )

            probabilities = (
                probabilities
                .squeeze(1)
                .cpu()
                .numpy()
            )

            for i, probability in enumerate(
                probabilities
            ):

                y, x = positions[
                    start + i
                ]

                probability_sum[
                    y:y + patch_size,
                    x:x + patch_size
                ] += probability

                count_map[
                    y:y + patch_size,
                    x:x + patch_size
                ] += 1

    # --------------------------------------------------
    # Combine patch predictions
    # --------------------------------------------------

    probability_map = (
        probability_sum /
        np.maximum(count_map, 1)
    )

    probability_map = probability_map[
        :working_height,
        :working_width
    ]

    # --------------------------------------------------
    # Resize prediction back to original dimensions
    # --------------------------------------------------

    if scale < 1.0:

        probability_map = cv2.resize(
            probability_map,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR
        )

    prediction_mask = (
        probability_map >= threshold
    ).astype(np.uint8)

    return (
        prediction_mask,
        probability_map
    )