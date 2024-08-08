import React from 'react';

const HowItWorks = () => (
  <div className="container my-4">
    <h1>How It Works</h1>
    <p>
      Our AI model utilizes advanced machine learning algorithms to analyze MRI scans and clinical data. Here's a step-by-step explanation of how it works:
    </p>
    <ol>
      <li>Data Collection: MRI scans and clinical data are collected from patients.</li>
      <li>Data Preprocessing: The collected data is cleaned and preprocessed to ensure it is ready for analysis.</li>
      <li>Model Training: The AI model is trained using historical data to learn patterns associated with Alzheimer's disease.</li>
      <li>Prediction: The trained model analyzes new MRI scans and clinical data to detect early signs of Alzheimer's.</li>
      <li>Results: The results are presented to healthcare professionals for further action.</li>
    </ol>
  </div>
);

export default HowItWorks;
