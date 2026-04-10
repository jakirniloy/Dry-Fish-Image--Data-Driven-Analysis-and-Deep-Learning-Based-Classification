# 🐟 Dry Fish Classification with Grad-CAM

A deep learning–based web application for classifying Bangladeshi dry fish species using image data. This project leverages **Convolutional Neural Networks (CNN)** with attention mechanisms and provides visual explanations using Grad-CAM.

---

## 📌 Project Overview

This project aims to automatically classify different types of dry fish using images. It is built with a hybrid CNN model and deployed using **Streamlit** for an interactive user interface.

The system allows users to:

* Upload a dry fish image
* Predict its class
* View prediction confidence
* Visualize important regions using Grad-CAM

---

## 🧠 Research Publication

This work is based on a published dataset and research article:

**Title:** *Dry Fish Image dataset: Data-driven analysis and deep learning-based classification*

**Authors:**
Amran Hossain, Md Jakir Hossain, Iffat Ara Arin, Md Jisan Mia, Abir Hasan, Tasnia Tasnim Rupoma, Md Nawab Yousuf Ali

**DOI:** https://doi.org/10.1016/j.dib.2026.112683

📖 Open Access under Creative Commons License

---

## ⚙️ Features

* 🐟 Multi-class dry fish classification (12 classes)
* 📊 Confidence score for predictions
* 🔥 Grad-CAM visualization for model interpretability
* 📈 Probability distribution chart
* 🎨 Clean and professional UI using Streamlit

---

## 🧾 Supported Classes

* Bashpata
* Chanda
* Chapila
* Chewa
* Churi
* Loitta
* Shukna Feuwa
* Shundori
* Chingri
* Kachki
* Narkeli
* Puti Chepa

---

## 🖼️ Image Preprocessing

* Resize to **224 × 224**
* Normalize pixel values (0–1)
* Convert image to array format

---

## 🔥 Grad-CAM Explanation

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize which parts of the image contribute most to the prediction.

This helps in:

* Model interpretability
* Trust and transparency
* Debugging misclassifications

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Streamlit

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py
├── requirements.txt
├── hybrid_cnn_attention_dryfish.h5
├── README.md
```

---

## 🚀 Deployment

This app can be deployed using:

* Streamlit Cloud
* Render
* Local server

⚠️ Recommended Python version: **3.10**

---

## 📊 Model Details

* Architecture: Hybrid CNN with Attention
* Input size: 224×224 RGB images
* Output: 12 dry fish classes
* Activation: Softmax

---

## ⚠️ Limitations

* Low confidence (<50%) images may not be classified correctly
* Performance depends on image quality
* Only supports predefined dry fish categories

---

## 🤝 Acknowledgements

We thank all contributors and researchers involved in building the dataset and model.

---

## 📬 Contact

For any queries or collaboration:

**Md Jakir Hossain** **mdjakirhossen13@gmail.com**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share with others!

## 🌐 Website

[🔗 Visit Live App](https://dry-fish-jhnnet.streamlit.app/)

![App Screenshot](https://github.com/jakirniloy/Dry-Fish-Image--Data-Driven-Analysis-and-Deep-Learning-Based-Classification/blob/main/Image/Screenshot%202026-04-10%20112908.png)
![App Screenshot](https://github.com/jakirniloy/Dry-Fish-Image--Data-Driven-Analysis-and-Deep-Learning-Based-Classification/blob/main/Image/Screenshot%202026-04-10%20112935.png)





