### AlzGuard - An AI-driven solution for early Alzheimer's detection

Early diagnosis of Alzheimer’s can delay the progression of the disease, and enhance the quality of life for patients and their families. On a larger scale, it can reduce healthcare costs and alleviate the strain on healthcare systems. With this goal in mind, we use a similar framework to The Stanford Institute for Human-Centered Artificial Intelligence. In this institute, a group of Stanford researchers were able to diagnose Alzheimer’s disease just by applying artificial intelligence (AI) with a high success rate. Their method was to use a trained neural network which could analyze PET/MRI scans and observe minuscule differences between a healthy brain and one that was facing nerve damage.

Our proposed innovation, AlzGuard, leverages the power of advanced artificial intelligence techniques and machine learning for the early diagnosis of Alzheimer’s disease. What sets AlzGuard apart is its holistic approach: we don't rely solely on MRI scans. Instead, we integrate comprehensive clinical data to deliver a more precise diagnosis. Our unique implementation of clinical data promises to transform the landscape of Alzheimer's diagnosis, offering hope to patients and their families. With AlzGuard, we're pioneering a brighter, more informed future in Alzheimer's care.

The application is built in Flutter with the AI model built with a convolutional neural network (CNN) to detect subtle changes in brain structure from MRIs. Our CNN was trained with an open-source dataset of MRI scans from both healthy individuals and Alzheimer’s patients. During the training process, the CNN used layers of artificial neurons to extract key image features. Then, through a process called convolution, the CNN applied filters to highlight these features and used pooling layers to reduce dimensionality while retaining key image information. Once trained, our CNN was capable of analyzing new MRI scans to assess the likelihood of a patient having Alzheimer’s. Logistic regression and meta-stacking were utilized to combine predictions from the two different models to predict Alzheimer's disease. The CNN provided high-dimensional, intricate insights from the MRI data, and the Random Forest Classifier made predictions based on patterns in clinical data which is also taken as input by the application.

### How to run website

Typing the commands in the terminal below will start the website. First, the libraries to run the Flask server must be installed:

```
pip install flask flask-cors
```

Next, we must run this command so that both the React and Flask server can run at the same time. This is necessary for the AI model to connect with the website.

```
npm install concurrently --save
```

Finally, to start the website, run the command below:

```
npm start
```
