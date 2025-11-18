# Legacy Workflow Guide (Original Notebooks)

This document outlines the manual workflow for running the original experiments found in the `experiments_archive` folder. 
**Note:** For the refactored, automated pipeline, please refer to the main `README.md` in the root directory.

---

## 1. Setup for New Data
1.  Copy all necessary folders from `new_task`.
2.  Place the data inside the `original_data` folder.

## 2. Data Preparation (`Prep.ipynb`)
1.  Open `Prep.ipynb`.
2.  Run the **1st block** to load functions.
3.  Run the **2nd block** to merge all data folders into a single directory.
4.  Run the **3rd block** to create a unified CSV file for the data.
5.  Run the **4th block** to convert units in the CSV from meters/radians to **millimeters/degrees**.
6.  **Medical Pipe Task Only:** Run the **5th and 6th blocks** to center the data (ROI).

## 3. Cropping (`Crop.ipynb`)
1.  Run the **1st block** several times to tune the cropping parameters visually.
2.  Run the **2nd block** to generate the dataset cropped 3 times (augmentation).
    * *Note:* You can change the number of crops in the loop parameter.

## 4. Splitting Data (`Split_train_test.ipynb`)
1.  Run the entire code to split the cropped data into `train` and `test` sets.
2.  The code generates a CSV file for each set.
3.  You can adjust the training set size using the parameter `test_size=1500`.

## 5. Labeling and Organization (`Label.ipynb`)
Run the code to organize data into folders based on labels (discretization):

* **Block 1:** Moves all test images to the `test angle` folder and renames files to their angle value.
* **Block 2:** Moves all test images to the `test rad` folder and renames files to their radius value.
* **Block 3:** Moves train images, classifies them into folders `00 – 15` (Classes), and renames them to the angle value.
* **Block 4:** Moves train images, classifies them into folders `0 – 5` (Classes), and renames them to the radius value.

## 6. Training RGB (`Train_rgb`)
1.  **Block 1:** Select the target variable: **Radius** or **Angle**.
2.  **Block 5:** Adjust training hyperparameters if necessary.
3.  **Blocks 6 & 7:** Visualize batches for Train and Test sets.
    * *Note:* The test set does not undergo augmentation.
4.  **Block 8:** Start training.
5.  **Block 9:** Save weights.
    * *Important:* Change the weight version number to avoid overwriting previous runs. Ensure you select Angle/Radius matching the training session.

## 7. Training Depth (`Train_depth`)
* Identical workflow to `Train_rgb`, with specific adjustments:
    * **Block 2:** Contains an additional line changing the input to **1 channel**.
    * **Block 3:** Includes a transformation to convert images to 1 channel (Grayscale) and excludes color augmentation for the training set.

## 8. Prediction (`Predict`)
* There are separate prediction scripts for each case: `RGB`, `Depth`, `Radius`, and `Angle`.
* At the end of each script, a graph showing the results will be generated.

## 9. Activation Maps (`Activation_map`)
* There are two separate codes: one for RGB and one for Depth.
    * *Note: The Depth code may require debugging.*
1.  **Setup:** Manually copy approximately **300 images** from the `Cropped` folder to the `test_1` folder located inside `activation_map` (separate folders for RGB/Depth cases).
2.  **Block 2:** Adjust the number of classes for the network.
3.  **Block 3:** Update the path to the weights file according to the version and network type (Angle/Radius).
4.  **Result:** The script generates a batch of images visualizing the task's activation maps.